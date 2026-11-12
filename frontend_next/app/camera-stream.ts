export type CameraStreamIdentity = { username: string; token: string };

const MAX_FRAME_BYTES = 4 * 1024 * 1024;
const MAX_HEADER_BYTES = 8192;
const decoder = new TextDecoder();

export function multipartBoundary(contentType: string): string {
  const match = /;\s*boundary\s*=\s*(?:"([^"\r\n]+)"|([^;\s]+))/i.exec(
    contentType,
  );
  const boundary = match?.[1] || match?.[2] || "";
  if (
    /[\r\n]/.test(contentType) ||
    !/^multipart\/x-mixed-replace\s*(?:;|$)/i.test(contentType) ||
    !boundary ||
    boundary.length > 70 ||
    /[^\x21-\x7e]/.test(boundary)
  ) {
    throw new Error("摄像头返回了无效的视频流格式");
  }
  return boundary;
}

export class MjpegParser {
  private buffer: Uint8Array = new Uint8Array();
  private boundary: Uint8Array;
  private phase: "boundary" | "headers" | "body" | "ended" = "boundary";
  private frameLength = 0;

  constructor(boundary: string) {
    if (!boundary || boundary.length > 70 || /[^\x21-\x7e]/.test(boundary)) {
      throw new Error("摄像头视频流分界无效");
    }
    this.boundary = new TextEncoder().encode(`--${boundary}`);
  }

  push(chunk: Uint8Array): Uint8Array[] {
    const frames: Uint8Array[] = [];
    for (let offset = 0; offset < chunk.length; offset += 65536) {
      const block = chunk.subarray(offset, offset + 65536);
      const combined = new Uint8Array(this.buffer.length + block.length);
      combined.set(this.buffer);
      combined.set(block, this.buffer.length);
      this.buffer = combined;
      this.parseAvailable(frames);
      if (this.buffer.length > MAX_FRAME_BYTES + MAX_HEADER_BYTES) {
        throw new Error("摄像头视频流缓冲超过限制");
      }
    }
    return frames;
  }

  private parseAvailable(frames: Uint8Array[]) {
    while (true) {
      if (this.phase === "ended") {
        if (this.buffer.some((byte) => ![9, 10, 13, 32].includes(byte))) {
          throw new Error("摄像头视频流结束后包含无效数据");
        }
        this.buffer = new Uint8Array();
        return;
      }
      if (this.phase === "boundary") {
        if (this.buffer[0] === 13 && this.buffer[1] === 10) {
          this.buffer = this.buffer.slice(2);
        }
        if (this.buffer.length < this.boundary.length + 2) return;
        if (this.boundary.some((byte, index) => byte !== this.buffer[index])) {
          throw new Error("摄像头视频流分界损坏");
        }
        const suffix = this.buffer.slice(
          this.boundary.length,
          this.boundary.length + 2,
        );
        if (suffix[0] === 45 && suffix[1] === 45) {
          this.phase = "ended";
        } else if (suffix[0] === 13 && suffix[1] === 10) {
          this.phase = "headers";
        } else {
          throw new Error("摄像头视频流分界损坏");
        }
        this.buffer = this.buffer.slice(this.boundary.length + 2);
        continue;
      }
      if (this.phase === "headers") {
        let headerEnd = -1;
        for (let index = 0; index <= this.buffer.length - 4; index++) {
          if (
            this.buffer[index] === 13 &&
            this.buffer[index + 1] === 10 &&
            this.buffer[index + 2] === 13 &&
            this.buffer[index + 3] === 10
          ) {
            headerEnd = index;
            break;
          }
        }
        if (headerEnd < 0) {
          if (this.buffer.length > MAX_HEADER_BYTES)
            throw new Error("摄像头视频流头部超过限制");
          return;
        }
        if (headerEnd > MAX_HEADER_BYTES)
          throw new Error("摄像头视频流头部超过限制");
        const headers = decoder.decode(this.buffer.slice(0, headerEnd));
        const lengths = [
          ...headers.matchAll(/^Content-Length:\s*(.*?)\s*$/gim),
        ];
        const contentType =
          /^Content-Type:\s*image\/jpeg\s*(?:;[^\r\n]*)?\r?$/im.test(headers);
        const length = lengths[0]?.[1] || "";
        this.frameLength = Number(length);
        if (
          !contentType ||
          lengths.length !== 1 ||
          !/^\d+$/.test(length) ||
          !Number.isSafeInteger(this.frameLength) ||
          this.frameLength < 4 ||
          this.frameLength > MAX_FRAME_BYTES
        ) {
          throw new Error("摄像头视频帧类型或长度无效");
        }
        this.buffer = this.buffer.slice(headerEnd + 4);
        this.phase = "body";
        continue;
      }
      if (this.buffer.length < this.frameLength) return;
      const frame = this.buffer.slice(0, this.frameLength);
      if (
        frame[0] !== 255 ||
        frame[1] !== 216 ||
        frame[frame.length - 2] !== 255 ||
        frame[frame.length - 1] !== 217
      ) {
        throw new Error("摄像头返回了损坏的视频帧");
      }
      frames.push(frame);
      this.buffer = this.buffer.slice(this.frameLength);
      this.phase = "boundary";
    }
  }

  finish() {
    if (this.phase === "ended") return;
    if (
      this.phase !== "boundary" ||
      this.buffer.some((byte) => ![10, 13].includes(byte))
    ) {
      throw new Error("摄像头视频流意外中断，帧数据不完整");
    }
  }
}

export async function receiveCameraStream(
  deviceId: string,
  apiBase: string,
  identity: CameraStreamIdentity,
  signal: AbortSignal,
  onFrame: (frame: Uint8Array) => void,
): Promise<void> {
  if (!identity.username) throw new Error("请先登录后查看摄像头实时预览");
  const url = new URL(
    `${apiBase}/api/devices/${encodeURIComponent(deviceId)}/stream`,
  );
  url.searchParams.set("username", identity.username);
  const headers = new Headers({ Accept: "multipart/x-mixed-replace" });
  if (identity.token) headers.set("Authorization", `Bearer ${identity.token}`);
  const response = await fetch(url.toString(), {
    headers,
    signal,
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      if (
        body &&
        typeof body === "object" &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      /* 非 JSON 错误保留状态码，避免遮蔽连接失败。 */
    }
    throw new Error(detail || `摄像头实时预览连接失败（${response.status}）`);
  }
  if (!response.body) throw new Error("浏览器无法读取摄像头视频流");
  const reader = response.body.getReader();
  const cancel = () => {
    void reader.cancel().catch(() => {});
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    const parser = new MjpegParser(
      multipartBoundary(response.headers.get("Content-Type") || ""),
    );
    if (signal.aborted) throw new Error("摄像头实时预览已取消");
    while (!signal.aborted) {
      const { value, done } = await reader.read();
      if (signal.aborted) throw new Error("摄像头实时预览已取消");
      if (done) {
        parser.finish();
        throw new Error("摄像头实时预览连接已结束，请重新连接");
      }
      for (const frame of parser.push(value)) {
        if (signal.aborted) return;
        onFrame(frame);
      }
    }
    throw new Error("摄像头实时预览已取消");
  } finally {
    signal.removeEventListener("abort", cancel);
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
