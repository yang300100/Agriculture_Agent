"""
病虫害识别模型 —— 使用演示
运行前请确保已执行: python setup_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model.inference import PestDiseaseDetector, PestDiseaseDetectorONNX, agent_tool_predict, agent_tool_schema

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "weights", "pest_disease_model.pth")
ONNX_PATH = os.path.join(os.path.dirname(__file__),
                          "weights", "pest_disease_model.onnx")


def demo_basic():
    """基础用法演示（PyTorch 后端）"""
    print("=" * 50)
    print("病虫害检测器 —— PyTorch 推理演示")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"\n模型文件不存在: {MODEL_PATH}")
        print("请先运行: python setup_model.py")
        return

    detector = PestDiseaseDetector(MODEL_PATH)
    print(f"模型已加载，设备: {detector.device}")

    demo_image = "test_leaf.jpg"
    if os.path.exists(demo_image):
        result = detector.predict(demo_image)
        print(f"\n预测结果:")
        print(f"  类别: {result['class']}")
        print(f"  置信度: {result['confidence']:.2%}")
        print(f"\nTop-5:")
        for i, item in enumerate(result["top_5"], 1):
            print(f"  {i}. {item['class']}: {item['confidence']:.2%}")
    else:
        print(f"\n(请将测试图像放到 {demo_image} 以查看预测结果)")


def demo_onnx():
    """ONNX 推理演示"""
    print("\n" + "=" * 50)
    print("病虫害检测器 —— ONNX 推理演示")
    print("=" * 50)

    if os.path.exists(ONNX_PATH):
        detector = PestDiseaseDetectorONNX(ONNX_PATH)
        print("ONNX 模型已加载")

        demo_image = "test_leaf.jpg"
        if os.path.exists(demo_image):
            result = detector.predict(demo_image)
            print(f"\nONNX 预测结果:")
            print(f"  类别: {result['class']}")
            print(f"  置信度: {result['confidence']:.2%}")
    else:
        print(f"\nONNX 模型不存在: {ONNX_PATH}")
        print("请先运行: python setup_model.py")


def demo_agent():
    """Agent 集成演示"""
    print("\n" + "=" * 50)
    print("Agent 工具函数演示")
    print("=" * 50)

    schema = agent_tool_schema()
    print("\nAgent Function Schema:")
    import json
    print(json.dumps(schema, ensure_ascii=False, indent=2))

    print("\n调用示例:")
    print('  result = agent_tool_predict("leaf_image.jpg")')
    print('  # result: {"class": "番茄早疫病", "confidence": 0.95, ...}')


if __name__ == "__main__":
    demo_basic()
    demo_onnx()
    demo_agent()
