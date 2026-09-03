"""
病虫害识别 —— Agent 接入 API
提供三种主流 Agent 框架的集成方式：
  1. LangChain Tool
  2. OpenAI Function Calling
  3. 独立 HTTP API（FastAPI）
"""

# ==================== 方式一：LangChain Tool ====================
# pip install langchain

def create_langchain_tool(model_path="weights/pest_disease_model.pth"):
    """
    创建 LangChain Tool 对象

    使用方式:
        from langchain.agents import initialize_agent
        tool = create_langchain_tool()
        agent = initialize_agent([tool], llm, agent="openai-tools")
        agent.invoke({"input": "识别 leaf.jpg 的病虫害"})
    """
    from langchain.tools import tool as lc_tool

    @lc_tool
    def identify_pest_disease(image_path: str) -> str:
        """
        识别农作物叶片图像中的病虫害类型。
        支持苹果、番茄、玉米、马铃薯等作物的38种病虫害。
        输入: 图像文件路径
        输出: 包含病虫害名称、置信度和Top-5预测的JSON字符串
        """
        import json
        from model.inference import agent_tool_predict
        result = agent_tool_predict(image_path)
        return json.dumps(result, ensure_ascii=False, indent=2)

    return identify_pest_disease


# ==================== 方式二：OpenAI Function Calling ====================

def get_openai_tool_definition():
    """
    返回 OpenAI Function Calling 工具定义

    使用方式:
        import openai
        tools = [get_openai_tool_definition()]
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[...],
            tools=tools,
        )
        # 当模型调用此工具时，执行 agent_tool_predict()
    """
    from model.inference import agent_tool_schema
    return agent_tool_schema()


def handle_openai_tool_call(arguments):
    """
    处理 OpenAI 返回的工具调用

    Args:
        arguments: {"image_path": "leaf.jpg"}

    Returns:
        病虫害识别结果的 JSON 字符串
    """
    import json
    from model.inference import agent_tool_predict

    image_path = arguments.get("image_path", "")
    result = agent_tool_predict(image_path)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==================== 方式三：FastAPI HTTP 服务 ====================
# pip install fastapi uvicorn python-multipart

def create_fastapi_app(model_path="weights/pest_disease_model.pth", use_onnx=False):
    """
    创建 FastAPI 应用

    启动方式:
        uvicorn agent_api:app --host 0.0.0.0 --port 8000

    API 端点:
        POST /predict        - 上传图像文件进行识别
        GET  /health         - 健康检查
        GET  /classes        - 获取所有类别列表

    Args:
        model_path: 模型权重路径 (.pth) 或 ONNX 模型路径 (.onnx)
        use_onnx: 是否使用 ONNX 推理后端
    """
    from fastapi import FastAPI, UploadFile, File
    from PIL import Image
    import io
    from model.config import CLASS_NAMES

    app = FastAPI(title="病虫害识别 API", version="1.0.0")

    if use_onnx:
        from model.inference import PestDiseaseDetectorONNX as Detector
    else:
        from model.inference import PestDiseaseDetector as Detector
    detector = Detector(model_path)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "ConvNeXt V2-Base", "classes": len(CLASS_NAMES)}

    @app.get("/classes")
    async def list_classes():
        return {"classes": CLASS_NAMES}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = detector.predict(image)
        return result

    return app


# ==================== 快速启动 ====================
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    print("病虫害识别 Agent API")
    print("=" * 40)
    print("\n三种集成方式已就绪:\n")
    print("1. LangChain Tool:    create_langchain_tool()")
    print("2. OpenAI Function:   get_openai_tool_definition()")
    print("3. FastAPI HTTP:      uvicorn agent_api:app --port 8000")
    print("\n确保已运行: python setup_model.py")
