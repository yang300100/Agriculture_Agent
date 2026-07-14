"""
摄像头拍照测试

测试 USB/IP/RTSP 摄像头的拍照功能。

依赖: pip install opencv-python numpy
运行:
  # USB 摄像头
  python hardware_examples/camera_capture_test.py

  # IP 摄像头
  python hardware_examples/camera_capture_test.py rtsp://admin:password@192.168.1.10/stream

  # 图片文件
  python hardware_examples/camera_capture_test.py test.jpg
"""

import sys
import os
import base64
import time

try:
    import cv2
    import numpy as np
except ImportError:
    print("❌ 请先安装 opencv-python: pip install opencv-python numpy")
    exit(1)


def test_camera(source, max_attempts=3):
    """测试单个摄像头源"""
    print(f"\n📷 测试摄像头: {source}")
    print("-" * 40)

    for attempt in range(max_attempts):
        cap = None
        try:
            print(f"   尝试 {attempt + 1}/{max_attempts}...")

            # 如果是数字字符串，转为 int (USB 摄像头索引)
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                print(f"   ❌ 无法打开摄像头 (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    time.sleep(2)
                continue

            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # 读取画面
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"   ❌ 读取画面失败")
                if attempt < max_attempts - 1:
                    time.sleep(1)
                continue

            # 获取画面信息
            height, width = frame.shape[:2]
            print(f"   ✅ 分辨率: {width}x{height}")
            print(f"   ✅ 画面大小: {frame.nbytes / 1024:.1f} KB")

            # 编码为 JPEG
            retval, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if retval:
                jpeg_bytes = jpeg.tobytes()
                print(f"   ✅ JPEG 编码: {len(jpeg_bytes) / 1024:.1f} KB")

                # base64 编码（用于 API 调用）
                b64 = base64.b64encode(jpeg_bytes).decode()
                print(f"   ✅ base64 长度: {len(b64)} 字符")

                # 保存测试图片
                test_dir = os.path.join(os.path.dirname(__file__), "test_output")
                os.makedirs(test_dir, exist_ok=True)
                filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(test_dir, filename)
                with open(filepath, "wb") as f:
                    f.write(jpeg_bytes)
                print(f"   💾 已保存: {filepath}")

            else:
                print(f"   ❌ JPEG 编码失败")

            return True

        except Exception as e:
            print(f"   ❌ 错误: {e}")
            if attempt < max_attempts - 1:
                time.sleep(1)
        finally:
            if cap is not None:
                cap.release()

    return False


def list_available_cameras():
    """列出所有可用的 USB 摄像头"""
    print("\n🔍 扫描可用摄像头...")
    available = []
    for i in range(5):  # 扫描前5个索引
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
                print(f"   ✅ 找到摄像头: index={i}")
            cap.release()
        else:
            cap.release()

    if not available:
        print("   ⚠ 未发现可用 USB 摄像头")
    return available


def main():
    print(f"\n{'='*50}")
    print(f"📸 摄像头拍照测试")
    print(f"{'='*50}")

    # 解析命令行参数
    if len(sys.argv) > 1:
        sources = sys.argv[1:]
    else:
        # 默认：扫描 USB 摄像头
        usb_cameras = list_available_cameras()
        if usb_cameras:
            sources = usb_cameras
        else:
            print("\n💡 用法:")
            print("   python hardware_examples/camera_capture_test.py            # 自动扫描 USB 摄像头")
            print("   python hardware_examples/camera_capture_test.py 0          # USB 摄像头 #0")
            print("   python hardware_examples/camera_capture_test.py rtsp://... # IP 摄像头")
            print("   python hardware_examples/camera_capture_test.py test.jpg   # 图片文件")
            sources = []

    if not sources:
        print("\n⚠ 未指定摄像头源，使用模拟测试...")
        _test_image_transform()
        return

    # 测试每个源
    success_count = 0
    for src in sources:
        if test_camera(src):
            success_count += 1

    print(f"\n{'='*50}")
    print(f"📊 结果: {success_count}/{len(sources)} 个摄像头测试成功")
    print(f"{'='*50}\n")

    if success_count == 0:
        print("💡 提示：如果没有物理摄像头，可以使用项目内置的 SimulatorDriver 进行虚拟测试。")
        print("   运行: python hardware_examples/test_integration.py")
        print("   或在对话中发送图片进行 AI 视觉分析。")


def _test_image_transform():
    """无摄像头时的模拟测试：生成测试图片"""
    print("   🎨 生成测试图片...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (100, 150, 50)  # 绿色背景
    cv2.putText(img, "Test Image", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    test_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(test_dir, exist_ok=True)
    filepath = os.path.join(test_dir, "test_generated.jpg")
    cv2.imwrite(filepath, img)
    print(f"   💾 测试图片已生成: {filepath}")


if __name__ == "__main__":
    main()
