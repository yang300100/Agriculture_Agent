"""
病虫害识别模型包
提供模型结构定义、推理接口和配置
"""
from .architecture import PestDiseaseClassifier, save_model_state, load_model_state, export_to_onnx
from .inference import PestDiseaseDetector, PestDiseaseDetectorONNX, agent_tool_predict, agent_tool_schema
from .config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE
