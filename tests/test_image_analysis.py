"""哥哥，图像回答必须忠实于本地观测并守住本轮回答范围。"""

import base64
import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from app.agent.agents.disease_agent import DiseaseAgent
from app.agent.nodes import image_analysis as image_node
from app.agent.state import AgentState
from models.base import ModelCapability, ModelInfo, ModelOutput, Prediction
from models.registry import ModelRegistry


@pytest.fixture
def image_case(monkeypatch):
    image = BytesIO()
    Image.new("RGB", (4, 4), "green").save(image, format="PNG")
    registry = ModelRegistry()
    registry._model_info["fixture-model"] = ModelInfo(
        model_id="fixture-model", model_name="离线测试模型", backend_name="torch",
        capability=ModelCapability.DISEASE_CLASSIFY, model_path="unused.pth",
        classes=["小麦-健康", "小麦-叶锈病", "小麦-茎锈病", "背景-无叶片"],
        metadata={"taxonomy": {"diseases": ["健康", "叶锈病", "茎锈病", "无叶片"]}},
    )
    raw_output = {
        "crop": {"class_name": "小麦", "confidence": 0.9123456789, "index": 2},
        "disease": {"class_name": "叶锈病", "confidence": 0.8912345678, "index": 2},
        "severity": {"class_name": "严重", "confidence": 0.8123456789, "index": 1},
        "scale_weights": [0.1, 0.2, 0.3, 0.4], "checkpoint_epoch": 7,
    }
    case = SimpleNamespace(
        output=ModelOutput(
            success=True, model_id="fixture-model",
            predictions=[Prediction("小麦-叶锈病", 0.9450713396072388, 1),
                         Prediction("小麦-茎锈病", 0.03123456789, 2)],
            inference_time_ms=123.456, raw_output=raw_output,
        ),
        registry=registry, prompts=[], weather_calls=[],
        response={
            "crop_type": "番茄", "growth_stage": "成熟期", "severity": "轻微",
            "overall_health": "良好", "description": "确诊为条锈病",
            "recommendations": [
                {"kind": "medication", "text": "按当地登记标签选用杀菌剂进行喷施。"},
                {"kind": "observation", "text": "记录病斑变化并补充叶片正反面照片。"},
            ],
            "urgency": "立即处理", "linked_action": "irrigate", "linked_params": {"duration": 9},
        },
        llm_error=None,
        state=AgentState(
            intent_type="image_analysis", has_image=True,
            image_data=base64.b64encode(image.getvalue()).decode("ascii"),
            user_question="请分析这张图片并给出防治建议。",
        ),
    )

    def infer(_executor, model_id, model_input):
        assert model_id == "fixture-model"
        assert model_input.image_bytes == image.getvalue()
        return case.output

    class OfflineLLM:
        def invoke(self, prompt):
            case.prompts.append(prompt)
            if case.llm_error:
                raise case.llm_error
            content = case.response if isinstance(case.response, str) else json.dumps(case.response, ensure_ascii=False)
            return SimpleNamespace(content=content)

    monkeypatch.setattr(image_node, "ENABLE_IMAGE_ANALYSIS", True)
    monkeypatch.setattr("core.model_registry_factory.resolve_inference_model", lambda _: (registry, "fixture-model"))
    monkeypatch.setattr("core.model_executor.ModelExecutor.infer_sync", infer)
    monkeypatch.setattr("app.agent.utils._get_llm", lambda: OfflineLLM())
    return case


def _run(case):
    return image_node.image_analysis_answer_node(image_node.image_analysis_node(case.state))


def test_local_multitask_observations_cannot_be_overwritten_by_advice(image_case):
    expected_raw = deepcopy(image_case.output.raw_output)
    result = _run(image_case).image_analysis_result
    assert result["raw_output"] == expected_raw
    assert result["predictions"] == [
        {"class_name": "小麦-叶锈病", "confidence": 0.9450713396072388, "index": 1},
        {"class_name": "小麦-茎锈病", "confidence": 0.03123456789, "index": 2},
    ]
    assert result["severity"] == "严重"
    assert result["severity_confidence"] == 0.8123456789
    assert result["detected_issues"][0]["severity"] == "严重"
    assert result["crop_type"] == "小麦"
    assert result["growth_stage"] in ("", "未知")
    assert result["overall_health"] in ("", "未知")
    assert "成熟期" not in image_case.state.final_answer
    assert "确诊为条锈病" not in image_case.state.final_answer


def test_advice_prompt_receives_question_full_observations_and_registry_coverage(image_case):
    image_case.state.user_question = "请说明识别依据，并提供观察建议。"
    _run(image_case)
    prompt = image_case.prompts[0]
    assert "请说明识别依据，并提供观察建议。" in prompt
    assert "0.8123456789" in prompt and "scale_weights" in prompt and "checkpoint_epoch" in prompt
    assert "小麦-茎锈病" in prompt and "背景-无叶片" in prompt
    assert "封闭" in prompt


def test_healthy_prediction_has_no_pest_disease_or_treatment_side_effects(image_case):
    image_case.output.predictions = [Prediction("小麦-健康", 0.99, 0)]
    image_case.output.raw_output["disease"]["class_name"] = "健康"
    state = _run(image_case)
    assert state.image_analysis_result["detected_issues"] == []
    assert state.image_analysis_result["recommendations"] == []
    assert state.skip_task_extraction and state.pending_action is None
    assert "虫害" not in state.final_answer and "杀菌剂" not in state.final_answer
    assert "小麦-健康" in state.final_answer and "未覆盖" in state.final_answer


def test_background_requests_another_photo_without_treatment(image_case):
    image_case.output.predictions = [Prediction("背景-无叶片", 0.99, 3)]
    image_case.output.raw_output = None
    state = _run(image_case)
    assert state.image_analysis_result["detected_issues"] == []
    assert state.image_analysis_result["uncertainty"]["requires_review"]
    assert "重拍" in state.final_answer
    assert state.skip_task_extraction and state.pending_action is None
    assert not state.image_analysis_result["recommendations"]


@pytest.mark.parametrize("predictions,reason", [
    ([Prediction("小麦-叶锈病", 0.69, 1), Prediction("小麦-茎锈病", 0.10, 2)], "low_confidence"),
    ([Prediction("小麦-叶锈病", 0.52, 1), Prediction("小麦-茎锈病", 0.47, 2)], "close_candidates"),
])
def test_uncertain_classification_is_review_only(image_case, predictions, reason):
    image_case.output.predictions = predictions
    state = _run(image_case)
    uncertainty = state.image_analysis_result["uncertainty"]
    assert uncertainty["requires_review"] and reason in uncertainty["reasons"]
    assert "复核" in state.final_answer
    assert state.image_analysis_result["recommendations"] == []
    assert state.image_analysis_result["linked_action"] == "none"
    assert state.skip_task_extraction and state.pending_action is None
    assert "杀菌剂" not in state.final_answer


def test_low_confidence_severity_is_preserved_but_blocks_treatment(image_case):
    image_case.output.raw_output["severity"]["confidence"] = 0.6584280133247375
    state = _run(image_case)
    assert state.image_analysis_result["severity_confidence"] == 0.6584280133247375
    assert "low_severity_confidence" in state.image_analysis_result["uncertainty"]["reasons"]
    assert state.skip_task_extraction and state.pending_action is None
    assert "严重" in state.final_answer and "复核" in state.final_answer


def test_class_coverage_comes_from_the_loaded_registry_not_an_invented_capability(image_case):
    image_case.registry._model_info["fixture-model"].classes = ["小麦__健康", "小麦__叶锈病", "小麦__茎锈病", "番茄__晚疫病"]
    image_case.state.user_question = "这张是不是条锈病或者赤霉病？"
    state = _run(image_case)
    coverage = state.image_analysis_result["coverage"]
    assert coverage["closed_set"] is True
    assert coverage["supported_classes"] == ["小麦-健康", "小麦-叶锈病", "小麦-茎锈病", "番茄-晚疫病"]
    assert coverage["crop_classes"] == ["小麦-健康", "小麦-叶锈病", "小麦-茎锈病"]
    assert "封闭" in state.final_answer and "未覆盖" in state.final_answer
    assert "条锈病" not in coverage["supported_classes"]
    assert "赤霉病" not in coverage["supported_classes"]


@pytest.mark.parametrize("image_data", ["%%%", base64.b64encode(b"not an image").decode("ascii"), ""])
def test_invalid_image_is_a_machine_readable_failure(image_case, image_data):
    image_case.state.image_data = image_data
    state = _run(image_case)
    assert state.error_code == "IMAGE_INVALID_INPUT"
    assert state.image_analysis_result["error"]
    assert state.skip_task_extraction and state.pending_action is None
    assert "分析出错" in state.final_answer


def test_empty_predictions_fail_instead_of_indexing_or_generating_advice(image_case):
    image_case.output.predictions = []
    state = _run(image_case)
    assert state.error_code == "IMAGE_NO_PREDICTIONS"
    assert state.skip_task_extraction and not image_case.prompts


def test_disabled_image_analysis_sets_a_failure_code(image_case, monkeypatch):
    monkeypatch.setattr(image_node, "ENABLE_IMAGE_ANALYSIS", False)
    state = _run(image_case)
    assert state.error_code == "IMAGE_DISABLED"
    assert state.skip_task_extraction and not image_case.prompts


def test_unavailable_model_sets_a_failure_code(image_case, monkeypatch):
    def unavailable(_model_id):
        raise RuntimeError("本地模型未加载")
    monkeypatch.setattr("core.model_registry_factory.resolve_inference_model", unavailable)
    state = _run(image_case)
    assert state.error_code == "IMAGE_MODEL_UNAVAILABLE"
    assert state.skip_task_extraction and state.pending_action is None


def test_failed_inference_keeps_upstream_error_code_and_blocks_followup(image_case):
    image_case.output = ModelOutput.error("fixture-model", "INFERENCE_ERROR")
    state = _run(image_case)
    assert state.error_code == "IMAGE_INFERENCE_FAILED"
    assert state.image_analysis_result["model_error_code"] == "INFERENCE_ERROR"
    assert state.skip_task_extraction and not image_case.prompts


@pytest.mark.parametrize("failure", [RuntimeError("offline"), "模型原始输出不是 JSON"])
def test_unavailable_advice_preserves_local_results_and_discloses_failure(image_case, failure):
    if isinstance(failure, Exception):
        image_case.llm_error = failure
    else:
        image_case.response = failure
    state = _run(image_case)
    assert state.error_code == ""
    assert state.image_analysis_result["raw_output"]["severity"]["class_name"] == "严重"
    assert "小麦-叶锈病" in state.final_answer and "严重" in state.final_answer
    assert "建议生成不可用" in state.final_answer
    assert state.image_analysis_result["recommendations"] == []
    assert state.skip_task_extraction and state.pending_action is None


def test_result_only_scope_excludes_advice_knowledge_and_automation(image_case):
    image_case.state.user_question = "只返回病名、置信度和模型原始严重度，不要给用药建议，也不要创建任务。"
    image_case.state.retrieved_docs = [{"page_content": "不要展示的知识片段：立即使用药剂。"}]
    state = _run(image_case)
    assert "小麦-叶锈病" in state.final_answer and "严重" in state.final_answer
    assert "81.23%" in state.final_answer
    assert "处理建议" not in state.final_answer and "知识片段" not in state.final_answer
    assert "杀菌剂" not in state.final_answer and "紧急程度" not in state.final_answer
    assert state.skip_task_extraction and state.pending_action is None
    assert state.image_analysis_result["response_policy"]["results_only"]


def test_no_medication_scope_cannot_leak_untrusted_medication_or_weather(image_case):
    image_case.state.user_question = "帮我识别，但不要给用药建议。"
    image_case.response["recommendations"] = [
        {"kind": "observation", "text": "观察期间立即喷施某杀菌剂。"},
        "使用某药剂防治。",
    ]
    state = _run(image_case)
    assert "杀菌剂" not in state.final_answer and "某药剂" not in state.final_answer
    assert state.image_analysis_result["linked_action"] == "none"
    assert state.pending_action is None
    assert not state.image_analysis_result["response_policy"]["allow_medication"]


def test_no_medication_uses_controlled_actions_instead_of_llm_free_text(image_case):
    image_case.state.user_question = "不要给用药建议，请提供观察与管理建议，也不要创建任务。"
    image_case.response["recommendations"] = [
        {"kind": "observation", "action": "record_symptoms", "text": "立即喷施某杀菌剂。"},
        {"kind": "nonchemical", "action": "check_ventilation", "text": "改用未知商品名。"},
        {"kind": "observation", "action": "invented_action", "text": "使用未登记药物。"},
    ]
    state = _run(image_case)
    assert state.image_analysis_result["recommendations"] == [
        "记录病斑范围、颜色及每日变化，供现场复核。",
        "检查种植环境通风和叶面凝露情况，结合现场条件调整管理。",
    ]
    assert image_case.prompts and "record_symptoms" in image_case.prompts[0]
    assert "商品名" not in state.final_answer and "杀菌剂" not in state.final_answer
    assert state.skip_task_extraction and state.pending_action is None


@pytest.mark.parametrize("question", [
    "识别图片但不要给用药建议也不要创建任务。",
    "请识别，不用药，不创建任务。",
    "请识别图片，不需要你推荐任何药剂，也不需要添加任何任务。",
])
def test_common_explicit_restrictions_are_enforced_without_punctuation_dependence(image_case, question):
    image_case.state.user_question = question
    state = _run(image_case)
    policy = state.image_analysis_result["response_policy"]
    assert not policy["allow_medication"] and not policy["allow_tasks"]
    assert "杀菌剂" not in state.final_answer
    assert state.skip_task_extraction and state.pending_action is None


@pytest.mark.parametrize("question", [
    "只提供观察与管理建议，不要用药，不要创建任务。",
    "仅提供非化学防治建议。",
])
def test_advice_only_request_is_not_misread_as_result_only(image_case, question):
    image_case.state.user_question = question
    image_case.response["recommendations"] = [{"kind": "observation", "action": "record_symptoms", "text": "记录症状。"}]
    state = _run(image_case)
    policy = state.image_analysis_result["response_policy"]
    assert not policy["results_only"] and not policy["allow_medication"]
    assert "记录病斑范围" in state.final_answer
    assert image_case.prompts


@pytest.mark.parametrize("question", [
    "请识别，不要给浇水和喷药建议。",
    "分析图片，不要给用药建议和任务建议。",
])
def test_prohibitions_apply_to_each_coordinated_target(image_case, question):
    image_case.state.user_question = question
    state = _run(image_case)
    assert not state.image_analysis_result["response_policy"]["allow_medication"]
    assert "杀菌剂" not in state.final_answer
    assert state.pending_action is None


@pytest.mark.parametrize("bad_advice", [
    {"recommendations": [{"kind": [], "text": "无效字段"}], "linked_action": "none"},
    {"recommendations": [], "linked_action": []},
])
def test_malformed_advice_fields_cannot_destroy_successful_local_observations(image_case, bad_advice):
    image_case.response = bad_advice
    state = _run(image_case)
    assert state.error_code == ""
    assert state.image_analysis_result["raw_output"]["severity"]["class_name"] == "严重"
    assert "小麦-叶锈病" in state.final_answer
    assert state.skip_task_extraction and state.pending_action is None


@pytest.mark.parametrize("kind", ["observation", "verification", "nonchemical", "general", "medication"])
def test_advice_cannot_reintroduce_unobserved_crop_stage_health_or_severity(image_case, kind):
    image_case.response["recommendations"] = [
        {"kind": kind, "text": "当前番茄处于苗期，病情轻微，整体健康良好。"},
    ]
    state = _run(image_case)
    assert "番茄" not in state.final_answer and "苗期" not in state.final_answer
    assert "轻微" not in state.final_answer and "健康良好" not in state.final_answer
    assert state.image_analysis_result["recommendations"] == []
    assert state.skip_task_extraction and state.pending_action is None
    assert "严重" in state.final_answer


@pytest.mark.parametrize("kind,action,expected", [
    ("observation", "record_symptoms", "记录病斑范围、颜色及每日变化，供现场复核。"),
    ("medication", "check_registered_products", "如现场复核确认需要药剂防治，请核对当地登记作物、病害与产品标签；杀菌剂、杀虫剂不能仅凭图像分类选定。"),
])
def test_default_advice_renders_controlled_action_instead_of_conflicting_free_text(image_case, kind, action, expected):
    image_case.response["recommendations"] = [
        {"kind": kind, "action": action, "text": "当前番茄处于苗期，病情轻微，整体健康良好。"},
    ]
    state = _run(image_case)
    assert state.image_analysis_result["recommendations"] == [expected]
    assert "番茄" not in state.final_answer and "苗期" not in state.final_answer
    assert "轻微" not in state.final_answer and "健康良好" not in state.final_answer
    assert "严重" in state.final_answer


def test_action_looking_medication_text_cannot_smuggle_current_image_facts(image_case):
    image_case.response["recommendations"] = [
        {"kind": "medication", "text": "建议给当前苗期番茄用药，病情轻微且整体健康良好。"},
    ]
    state = _run(image_case)
    assert "苗期" not in state.final_answer and "番茄" not in state.final_answer
    assert "轻微" not in state.final_answer and "健康良好" not in state.final_answer


def test_linked_parameters_drop_free_text_and_nested_fields(image_case):
    image_case.response["recommendations"] = [{"kind": "observation", "action": "record_symptoms"}]
    image_case.response["linked_params"] = {
        "duration": 9,
        "reason": "当前番茄处于苗期，病情轻微，整体健康良好。",
        "nested": {"description": "未知结论"},
        "target_temp": 22,
    }
    state = _run(image_case)
    assert state.image_analysis_result["linked_params"] == {"duration": 9}
    assert state.pending_action["params"] == {"duration": 9}
    assert "番茄" not in state.final_answer and "苗期" not in state.final_answer
    assert "轻微" not in state.final_answer and "健康良好" not in state.final_answer
    assert "nested" not in state.final_answer and "target_temp" not in state.final_answer


@pytest.mark.parametrize("params", [
    {}, {"duration": True}, {"duration": False}, {"duration": "9"},
    {"duration": float("nan")}, {"duration": float("inf")},
    {"duration": -1}, {"duration": 0}, {"duration": 0.5}, {"duration": 121},
    {"duration": {"value": 9}}, {"duration": [9]},
    {"reason": "当前番茄处于苗期。"}, [9], "9",
])
def test_invalid_duration_or_parameter_shape_cannot_create_device_action(image_case, params):
    image_case.response["recommendations"] = [{"kind": "observation", "action": "record_symptoms"}]
    image_case.response["linked_params"] = params
    state = _run(image_case)
    assert state.pending_action is None
    assert state.image_analysis_result["linked_action"] == "none"
    assert state.image_analysis_result["linked_params"] == {}
    assert "建议设备操作" not in state.final_answer
    assert state.error_code == ""


@pytest.mark.parametrize("action,duration", [("irrigate", 120), ("ventilate", 120), ("light", 720), ("heat", 240), ("shade", 240)])
def test_device_duration_accepts_existing_schema_limit_but_rejects_overrun(image_case, action, duration):
    image_case.response["recommendations"] = [{"kind": "observation", "action": "record_symptoms"}]
    image_case.response["linked_action"] = action
    image_case.response["linked_params"] = {"duration": duration}
    state = _run(image_case)
    assert state.pending_action["params"] == {"duration": duration}
    image_case.state = AgentState(
        has_image=True, image_data=base64.b64encode(b"invalid").decode("ascii"), intent_type="image_analysis",
    )
    image_case.response["linked_params"] = {"duration": duration + 1}
    # 哥哥，直接复用有效的本地观测，只单独核对最终渲染这道边界。
    image_case.state.image_analysis_result = deepcopy(state.image_analysis_result)
    image_case.state.image_analysis_result["linked_params"] = {"duration": duration + 1}
    image_node.image_analysis_answer_node(image_case.state)
    assert image_case.state.pending_action is None
    assert "建议设备操作" not in image_case.state.final_answer


def test_user_facing_prediction_labels_do_not_expose_task_head_terms(image_case):
    state = _run(image_case)
    assert "作物预测" in state.final_answer and "病害预测" in state.final_answer
    assert "作物头" not in state.final_answer and "病害头" not in state.final_answer
    image_case.output.raw_output["disease"]["confidence"] = 0.4
    state.image_analysis_result["uncertainty"]["requires_review"] = True
    image_node.image_analysis_answer_node(state)
    assert "独立预测" in state.final_answer and "任务头" not in state.final_answer


def test_image_medication_advice_cannot_append_free_text_weather_facts(image_case, monkeypatch):
    def weather(_agent, intent, state):
        image_case.weather_calls.append(intent)
        return "适合喷药。当前番茄处于苗期，病情轻微，整体健康良好。"
    monkeypatch.setattr(DiseaseAgent, "call_colleague", weather)
    state = DiseaseAgent().invoke(image_case.state)
    assert "番茄" not in state.final_answer and "苗期" not in state.final_answer
    assert "轻微" not in state.final_answer and "健康良好" not in state.final_answer
    assert "当地最新天气" in state.final_answer
    assert image_case.weather_calls == []


def test_exact_confidence_threshold_allows_supported_advice(image_case):
    image_case.output.predictions[0].confidence = 0.7
    state = _run(image_case)
    assert not state.image_analysis_result["uncertainty"]["requires_review"]
    assert "杀菌剂" in state.final_answer


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_model_confidence_is_a_failure_instead_of_an_action(image_case, confidence):
    image_case.output.predictions[0].confidence = confidence
    state = _run(image_case)
    assert state.error_code == "IMAGE_INVALID_OUTPUT"
    assert state.skip_task_extraction and state.pending_action is None


def test_legacy_classifier_does_not_gain_a_made_up_severity(image_case):
    image_case.output.raw_output = None
    state = _run(image_case)
    assert state.image_analysis_result["severity"] == ""
    assert state.image_analysis_result["severity_confidence"] is None
    assert "模型原始严重度**: 未提供" in state.final_answer
    assert "轻微" not in state.final_answer


def test_no_task_scope_is_deterministic_even_when_advice_suggests_actions(image_case):
    image_case.state.user_question = "分析病害并提供建议，但不要创建任务。"
    state = _run(image_case)
    assert state.skip_task_extraction and state.pending_action is None
    assert state.image_analysis_result["recommendations"]
    assert not state.image_analysis_result["response_policy"]["allow_tasks"]


@pytest.mark.parametrize("question", [
    "不要只返回识别结果，请给出用药建议并创建任务。",
    "不是不需要用药建议，请提供防治建议并创建任务。",
    "不要仅解释病名，请给用药建议并创建任务。",
])
def test_negated_scope_words_do_not_reverse_positive_request(image_case, question):
    image_case.state.user_question = question
    state = _run(image_case)
    policy = state.image_analysis_result["response_policy"]
    assert not policy["results_only"]
    assert policy["allow_medication"] and policy["allow_tasks"]
    assert "杀菌剂" in state.final_answer
    assert not state.skip_task_extraction


@pytest.mark.parametrize("kind", ["failed", "healthy", "review", "result_only", "no_medication"])
def test_disease_agent_does_not_fetch_weather_for_ineligible_image_results(image_case, monkeypatch, kind):
    if kind == "failed":
        image_case.output = ModelOutput.error("fixture-model", "INFERENCE_ERROR")
    elif kind == "healthy":
        image_case.output.predictions = [Prediction("小麦-健康", 0.99, 0)]
    elif kind == "review":
        image_case.output.predictions[0].confidence = 0.2
    elif kind == "result_only":
        image_case.state.user_question = "只返回识别结果。"
    else:
        image_case.state.user_question = "识别病名，不要给用药建议。"
    def weather(_agent, intent, state):
        image_case.weather_calls.append(intent)
        return "适合喷药，风力低。"
    monkeypatch.setattr(DiseaseAgent, "call_colleague", weather)
    state = DiseaseAgent().invoke(image_case.state)
    assert image_case.weather_calls == []
    assert "施药气象" not in state.final_answer
