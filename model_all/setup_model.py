"""
一键生成模型文件 + ONNX 导出
在 Python 环境中运行一次即可：
    python setup_model.py

将自动：
1. 下载 ConvNeXt V2-Base 预训练权重（timm）
2. 构建病虫害分类模型
3. 保存为 state_dict .pth 文件（安全格式）
4. 导出 ONNX 模型供快速推理
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from model.architecture import PestDiseaseClassifier, save_model_state, export_to_onnx
from model.config import NUM_CLASSES, DROPOUT_RATE, FREEZE_BACKBONE, IMAGE_SIZE

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "weights")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pest_disease_model.pth")
ONNX_PATH = os.path.join(OUTPUT_DIR, "pest_disease_model.onnx")


def main():
    print("=" * 50)
    print("病虫害识别模型 —— 一键构建 + ONNX 导出")
    print("=" * 50)

    # 1. 构建模型（自动下载 timm 预训练权重）
    print("\n[1/5] 构建模型并下载预训练权重...")
    model = PestDiseaseClassifier(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=FREEZE_BACKBONE,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  骨干网络: ConvNeXt V2-Base (timm pretrained, FCMAE 自监督)")
    print(f"  特征维度: {model.feature_dim}")
    print(f"  分类类别: {NUM_CLASSES} 类")
    print(f"  总参数量:   {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # 2. 保存 state_dict
    print(f"\n[2/5] 保存模型权重到: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_model_state(model, OUTPUT_PATH)
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  文件大小: {file_size:.1f} MB")

    # 3. 验证 .pth 加载
    print(f"\n[3/5] 验证 state_dict 加载...")
    from model.architecture import load_model_state as load_state
    loaded = load_state(OUTPUT_PATH)
    loaded.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        out = loaded(dummy)
    print(f"  输入:  (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE})")
    print(f"  输出:  {tuple(out.shape)}  (预期: (1, {NUM_CLASSES}))")
    print(f"  state_dict 加载验证通过！")

    # 4. 导出 ONNX
    print(f"\n[4/5] 导出 ONNX 模型到: {ONNX_PATH}")
    export_to_onnx(loaded, ONNX_PATH, image_size=IMAGE_SIZE)
    onnx_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"  文件大小: {onnx_size:.1f} MB")

    # 5. 验证 ONNX 推理
    print(f"\n[5/5] 验证 ONNX 推理...")
    from model.inference import PestDiseaseDetectorONNX
    onnx_detector = PestDiseaseDetectorONNX(ONNX_PATH)
    import numpy as np
    ort_inputs = {"input": dummy.numpy()}
    ort_out = onnx_detector.session.run(None, ort_inputs)[0]
    print(f"  ONNX 输出: {tuple(ort_out.shape)}  (预期: (1, {NUM_CLASSES}))")
    print(f"  ONNX 推理验证通过！")

    print(f"\n{'=' * 50}")
    print(f"模型文件已就绪:")
    print(f"  PyTorch: {OUTPUT_PATH}")
    print(f"  ONNX:    {ONNX_PATH}")
    print(f"\n使用方式:")
    print(f"  PyTorch 推理: detector = PestDiseaseDetector('{OUTPUT_PATH}')")
    print(f"  ONNX 推理:   detector = PestDiseaseDetectorONNX('{ONNX_PATH}')")
    print(f"  Agent 工具:   agent_tool_predict('leaf.jpg')")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
