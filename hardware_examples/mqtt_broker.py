"""
轻量 MQTT Broker — 开发/测试用，无需安装 mosquitto

基于 asyncio 实现基本的 MQTT 3.1.1 CONNECT / SUBSCRIBE / PUBLISH。
仅支持 QoS 0，无认证，无 retain/will。

启动: python hardware_examples/mqtt_broker.py [--port 1883]
"""

import asyncio
import struct
import sys

HOST = "127.0.0.1"
PORT = 1883

# MQTT 数据包类型
CONNECT = 1
CONNACK = 2
PUBLISH = 3
SUBSCRIBE = 8
SUBACK = 9
PINGREQ = 12
PINGRESP = 13
DISCONNECT = 14

# 主题订阅表: {topic_pattern: [writer1, writer2, ...]}
subscriptions = {}
# 通配符订阅: [(pattern, writer), ...]
wildcard_subs = []


def match_topic(pattern, topic):
    """简单通配符匹配: + 匹配单层, # 匹配多层"""
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")
    for i, pp in enumerate(pattern_parts):
        if pp == "#":
            return True
        if i >= len(topic_parts):
            return False
        if pp == "+":
            continue
        if pp != topic_parts[i]:
            return False
    return len(pattern_parts) == len(topic_parts)


def make_connack():
    return bytes([0x20, 0x02, 0x00, 0x00])


def make_suback(packet_id):
    pid = struct.pack(">H", packet_id)
    # QoS 0 for each subscription
    remaining = 2 + 1
    return bytes([0x90, remaining]) + pid + bytes([0x00])


def make_publish(topic, payload, packet_id=None):
    topic_bytes = topic.encode("utf-8")
    flags = 0x30  # QoS 0
    remaining = 2 + len(topic_bytes) + len(payload)
    header = bytes([flags, remaining]) + struct.pack(">H", len(topic_bytes)) + topic_bytes
    return header + payload


def parse_remaining_length(data, offset):
    """解析 MQTT 变长整数"""
    multiplier = 1
    value = 0
    while offset < len(data):
        byte = data[offset]
        value += (byte & 0x7F) * multiplier
        offset += 1
        if (byte & 0x80) == 0:
            break
        multiplier *= 128
    return value, offset


def parse_connect(data):
    """解析 CONNECT 包，返回 client_id"""
    try:
        # 跳过 protocol name
        proto_len = struct.unpack(">H", data[2:4])[0]
        pos = 4 + proto_len  # protocol level + flags + keepalive
        pos += 3
        # 读 client_id
        client_len = struct.unpack(">H", data[pos:pos + 2])[0]
        pos += 2
        client_id = data[pos:pos + client_len].decode("utf-8")
        return client_id
    except Exception:
        return "unknown"


def parse_subscribe(data, offset):
    """解析 SUBSCRIBE 包，返回 (packet_id, topics)"""
    packet_id = struct.unpack(">H", data[offset:offset + 2])[0]
    offset += 2
    topics = []
    while offset < len(data):
        topic_len = struct.unpack(">H", data[offset:offset + 2])[0]
        offset += 2
        topic = data[offset:offset + topic_len].decode("utf-8")
        offset += topic_len + 1  # skip QoS byte
        topics.append(topic)
    return packet_id, topics


async def handle_client(reader, writer):
    addr = writer.get_extra_info("peername")
    client_id = str(addr)
    buf = b""

    try:
        while True:
            data = await asyncio.wait_for(reader.read(4096), timeout=120)
            if not data:
                break
            buf += data

            while len(buf) >= 2:
                packet_type = (buf[0] & 0xF0) >> 4
                if packet_type not in (CONNECT, SUBSCRIBE, PUBLISH, PINGREQ, DISCONNECT):
                    break

                try:
                    remaining, pos = parse_remaining_length(buf, 1)
                except Exception:
                    break

                total = pos + remaining
                if len(buf) < total:
                    break  # 数据不完整，等待更多数据

                packet = buf[:total]
                buf = buf[total:]

                if packet_type == CONNECT:
                    client_id = parse_connect(packet)
                    writer.write(make_connack())
                    await writer.drain()
                    print(f"[Broker] 客户端连接: {client_id}")

                elif packet_type == SUBSCRIBE:
                    packet_id, topics = parse_subscribe(packet, pos)
                    for topic in topics:
                        if "#" in topic or "+" in topic:
                            wildcard_subs.append((topic, writer))
                        else:
                            subscriptions.setdefault(topic, []).append(writer)
                    writer.write(make_suback(packet_id))
                    await writer.drain()
                    print(f"[Broker] {client_id} 订阅: {topics}")

                elif packet_type == PUBLISH:
                    topic_len = struct.unpack(">H", packet[pos:pos + 2])[0]
                    topic = packet[pos + 2:pos + 2 + topic_len].decode("utf-8")
                    payload = packet[pos + 2 + topic_len:]
                    # 分发给订阅者
                    targets = list(subscriptions.get(topic, []))
                    for pattern, w in wildcard_subs:
                        if match_topic(pattern, topic):
                            targets.append(w)
                    pub_pkt = make_publish(topic, payload)
                    for w in targets:
                        if w != writer:  # 不发给自己
                            try:
                                w.write(pub_pkt)
                                await w.drain()
                            except Exception:
                                pass

                elif packet_type == PINGREQ:
                    writer.write(bytes([0xD0, 0x00]))
                    await writer.drain()

                elif packet_type == DISCONNECT:
                    break

    except (asyncio.TimeoutError, ConnectionError, OSError):
        pass
    finally:
        # 清理订阅
        for topic in list(subscriptions):
            subscriptions[topic] = [w for w in subscriptions[topic] if w != writer]
            if not subscriptions[topic]:
                del subscriptions[topic]
        wildcard_subs[:] = [(p, w) for p, w in wildcard_subs if w != writer]
        try:
            writer.close()
        except Exception:
            pass
        print(f"[Broker] 客户端断开: {client_id}")


async def main():
    server = await asyncio.start_server(handle_client, HOST, PORT)
    print(f"[MQTT Broker] 启动: {HOST}:{PORT}")
    print("  支持 CONNECT / SUBSCRIBE / PUBLISH (QoS 0)")
    print("  通配符: + (单层) / # (多层)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()
    PORT = args.port
    HOST = args.host
    asyncio.run(main())
