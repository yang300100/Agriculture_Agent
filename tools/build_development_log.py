from pathlib import Path
from datetime import date
from shutil import copy2

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepTogether,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PDF_OUT = OUT / "pdf"
POSTER_OUT = OUT / "poster"
GENERATED_POSTER = Path(r"C:\Users\User\.codex\generated_images\01a03dd8-1328-70c0-94f6-0161aff3df14\exec-608305d3-994f-48a1-b3d0-32292dc9a31e.png")
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")


def register_fonts():
    pdfmetrics.registerFont(TTFont("MicrosoftYaHei", str(FONT_PATH), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("MicrosoftYaHeiBold", str(FONT_PATH), subfontIndex=1))


def build_poster() -> Path:
    POSTER_OUT.mkdir(parents=True, exist_ok=True)
    output = POSTER_OUT / "he_shu_zhi_nong_product_poster_illustrated.png"
    image = Image.open(GENERATED_POSTER).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    w, h = image.size
    title_font = ImageFont.truetype(str(FONT_PATH), 86, index=1)
    subtitle_font = ImageFont.truetype(str(FONT_PATH), 34, index=0)
    feature_font = ImageFont.truetype(str(FONT_PATH), 27, index=0)
    # 顶部白色天空是为了给中文标题预留的干净信息区。
    draw.text((88, 88), "禾枢智农", font=title_font, fill=(19, 76, 53, 255))
    draw.text((94, 194), "多智能体智慧种植决策与管控平台", font=subtitle_font, fill=(38, 92, 73, 255))
    draw.rounded_rectangle((88, 300, 1012, 370), radius=20, fill=(20, 96, 82, 210))
    draw.text((118, 318), "环境感知  ·  智能分析  ·  方案生成  ·  设备执行  ·  结果记录", font=feature_font, fill=(255, 255, 255, 255))
    footer = "多智能体协同 ｜ 视觉巡检 ｜ IoT 安全联动"
    footer_font = ImageFont.truetype(str(FONT_PATH), 31, index=0)
    box = draw.textbbox((0, 0), footer, font=footer_font)
    draw.text(((w - (box[2] - box[0])) / 2, h - 170), footer, font=footer_font, fill=(230, 255, 240, 255))
    image.save(output, "PNG", optimize=True)
    return output


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="MicrosoftYaHeiBold", fontSize=26, leading=35, textColor=colors.HexColor("#174C35"), alignment=TA_CENTER, spaceAfter=13),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName="MicrosoftYaHei", fontSize=11, leading=18, textColor=colors.HexColor("#52665B"), alignment=TA_CENTER),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="MicrosoftYaHeiBold", fontSize=17, leading=25, textColor=colors.HexColor("#174C35"), spaceBefore=8, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="MicrosoftYaHeiBold", fontSize=12, leading=19, textColor=colors.HexColor("#286348"), spaceBefore=8, spaceAfter=5),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="MicrosoftYaHei", fontSize=9.3, leading=16, spaceAfter=5),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName="MicrosoftYaHei", fontSize=7.7, leading=12, textColor=colors.HexColor("#52665B")),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="MicrosoftYaHei", fontSize=8, leading=12, textColor=colors.HexColor("#52665B"), alignment=TA_CENTER),
    }


def P(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(rows, widths, st):
    data = [[P(cell, st["small"]) for cell in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEEE3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#174C35")),
        ("FONTNAME", (0, 0), (-1, 0), "MicrosoftYaHeiBold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AFC8B8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FBF8")]),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("MicrosoftYaHei", 7.5)
    canvas.setFillColor(colors.HexColor("#52665B"))
    canvas.drawString(1.6 * cm, 1.15 * cm, "禾枢智农项目开发日志")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.15 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf(poster: Path):
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    output = PDF_OUT / "禾枢智农_项目开发日志_对外展示版_2026-08-26.pdf"
    st = styles()
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.45*cm, bottomMargin=1.6*cm)
    story = []
    story += [Spacer(1, 1.0*cm), P("禾枢智农", st["title"]), P("多智能体智慧种植决策与管控平台", st["subtitle"]), Spacer(1, .35*cm)]
    story += [RLImage(str(poster), width=10.2*cm, height=15.3*cm), Spacer(1, .35*cm)]
    story += [P("项目开发日志", st["h2"]), P("编制日期：2026 年 8 月 26 日　|　项目代码库：Agriculture_Agent", st["subtitle"]), PageBreak()]

    story += [P("一、项目简介", st["h1"])]
    story += [P("禾枢智农是一套面向种植场景的多智能体智慧种植决策与管控平台。平台围绕“环境感知—智能分析—方案生成—设备执行—结果记录”构建工作闭环，覆盖农业咨询、种植规划、农事管理、视觉巡检、设备控制与生产数据管理等核心环节。", st["body"])]
    story += [table([
        ["建设模块", "核心内容", "项目价值"],
        ["智能决策", "七类专业智能体协同调度，支持农业问答、种植规划与农事任务编排", "让种植决策更贴近实际生产流程"],
        ["生产管理", "地块、进度、提醒、财务、政策与作物知识一体化管理", "形成可持续沉淀的生产记录"],
        ["视觉巡检", "摄像头定时采集与作物健康分析", "提升日常巡检效率与问题发现速度"],
        ["设备联动", "灌溉、施肥、通风、补光等设备控制与规则自动化", "建立决策到执行的农业物联网闭环"],
    ], [2.4*cm, 8.2*cm, 6.3*cm], st)]
    story += [P("项目定位：以“环境感知—智能分析—方案生成—设备执行—结果记录”为闭环，结合多智能体调度、农业知识检索、视觉诊断、生产管理与可控 IoT 设备联动。", st["body"]), PageBreak()]

    story += [P("二、全流程关键时间线", st["h1"])]
    story += [table([
        ["日期 / 阶段", "关键节点", "真实佐证与结果"],
        ["2026-06-19\n自主决策原型", "提出自主决策配置；建立 AutonomousFarmManager 的数据收集、LLM 决策、执行编排与规则兜底。", "连续 6 条 feat/chore 提交：20de736e 至 007b5c05。"],
        ["2026-06-19\n交互可用性迭代", "围绕右侧可折叠导航、登录状态保持与 Streamlit DOM/CSS 兼容开展多轮修复。", "提交链 ab531e01 至 2317896a，记录了 iframe、radio、CSS 定位等多种方案的试验与回退。"],
        ["2026-06-19\n设备执行闭环", "修复设备执行日志路径，补齐 need_confirm/rejected 等决策状态写入。", "提交 66604259、b63d6e10。"],
        ["2026-07-14\n持久化架构升级", "从 JSON 迁移到 SQLAlchemy ORM + Repository，随后将 10 个模块切换为 SQLite 优先存储。", "0f8b6d1a、d0cd78fc 等提交；含迁移脚本与双写过渡。"],
        ["2026-07-14\n模型接入", "设计 ONNX + Torch 双后端，图像分析由本地深度学习模型与 LLM 增强协同。", "72daeeab、f5964890、04412cf。"],
        ["2026-07-14\n协议仿真", "建成 HTTP、MQTT、Modbus 三协议共享状态模拟器，后续补齐 CLI 指令与参数化设备卡片。", "45ad684c 至 91ff5c1a；最终 v5.0 完整协议栈提交 647dffc0。"],
        ["2026-07-14\n稳定性收敛", "集中修复连接泄漏、日期转换、设备能力误判、失败动作配额、异步与测试适配。", "88de2a0e、96b953d2、77fdb29c、f5d4e37e 等提交。"],
        ["2026-08-01 至 08-22\n验证与加固", "扩展 CoAP/OPC UA；地块—作业区—设备关联；安全策略、路径隔离与摄像头存储修复。", "阶段性回归结果为 159、170、183、185 项后端测试通过，并有 33 项前端测试记录。"],
    ], [2.8*cm, 7.0*cm, 7.1*cm], st), PageBreak()]

    story += [P("三、方案设计与架构决策", st["h1"])]
    story += [P("1. 多智能体设计：以调度中心解析复合意图并将请求分派给种植、病虫害、气象、财务、农事、设备、作物监测七个专业 Agent。对病虫害—气象—设备这类跨域问题，允许 Agent 互调并合并回答。", st["body"])]
    story += [P("2. 安全执行设计：设备动作经过规则边界与安全策略，采用待确认、执行中、执行成功/失败等状态留痕，避免模型输出绕开设备执行器直接控制硬件。", st["body"])]
    story += [P("3. 存储演进：初期 JSON 存储便于快速落地；随着多业务模块增加，切换到 SQLite、ORM、Repository 与迁移脚本，减少结构化数据的分散维护成本。", st["body"])]
    story += [P("4. 软硬件联动：以统一驱动注册中心兼容 Simulator、HTTP、MQTT、Modbus，并在后续扩展 CoAP、OPC UA；终端模拟器保持共享状态，便于无真实硬件条件下验证控制链路。", st["body"])]
    for fn, cap in [("01_multi_agent_dispatch.png", "图 1：多智能体调度设计"), ("03_device_safety_filter.png", "图 2：设备安全过滤设计"), ("07_device_driver_tree.png", "图 3：设备驱动树")]:
        path = ROOT / "report_images" / fn
        if path.exists():
            story += [RLImage(str(path), width=13.5*cm, height=7.6*cm), P(cap, st["caption"]), Spacer(1, .2*cm)]
    story.append(PageBreak())

    story += [P("四、实验测试与数据记录", st["h1"])]
    story += [P("项目围绕自主巡检、软硬件协议、应用服务、前端交互与安全控制持续开展测试与回归，测试用例随功能迭代逐步完善。", st["body"])]
    story += [table([
        ["阶段", "验证内容", "记录结果", "结论边界"],
        ["2026-07-29", "调度、自主巡检、模型注册及安全检查", "112 项测试通过", "完成核心功能回归。"],
        ["2026-08-01", "HTTP、MQTT、Modbus TCP、CoAP、OPC UA 协议链路", "集成检查通过", "完成多协议设备接入验证。"],
        ["2026-08-04", "分区控制、自主巡检、安全门与规则持久化", "159 项测试通过", "覆盖地块/作业区/设备及安全边界。"],
        ["2026-08-13", "应用加固与前端回归", "170 项 Python + 33 项前端测试通过", "完成前后端协同回归。"],
        ["2026-08-18", "启动健康、待处理设备操作、API 语义、RAG 复用", "183 项 Python + 33 项前端测试通过", "提升应用服务稳定性。"],
        ["2026-08-22", "路径安全、存储隔离、摄像头用户隔离与持久化", "185 项 Python + 33 项前端测试通过", "完善数据安全与访问控制。"],
    ], [2.2*cm, 5.7*cm, 3.2*cm, 5.8*cm], st)]
    story += [P("通过分阶段测试，平台的智能调度、设备控制、数据管理与交互页面形成了稳定协同的运行流程。", st["body"]), PageBreak()]

    story += [P("五、问题排查与解决纪要（开发复盘）", st["h1"])]
    story += [table([
        ["问题", "排查过程 / 决策", "解决结果"],
        ["导航切换导致登录丢失", "依次试验按钮、radio、CSS 精确选择、iframe；结合提交记录保留兼容性更高的 radio 方案。", "导航状态与页面重载问题得到收敛，提交历史保留了完整试验过程。"],
        ["数据库迁移后多类测试失败", "围绕 Date 转换、连接池、传感器初始化、异步测试和模拟器适配逐项修复。", "一次集中修复记录显示全部 19 个当时失败测试被修复。"],
        ["设备能力误判与连接泄漏", "检查驱动注册与能力解析的异常路径，补充关闭 registry、空能力安全处理。", "避免空结果被默认授权为灌溉能力，并降低连接遗留风险。"],
        ["路径穿越及跨用户存储风险", "限制报告/标识符/作物路径组件，统一 DATA_STORAGE_DIR，并检查摄像头捕获二次落盘。", "阶段记录显示路径穿越探针与用户隔离验证通过。"],
        ["跨模块功能协同", "对调度、设备、存储和界面操作进行联动检查。", "形成从任务理解到设备执行的完整业务流程。"],
    ], [3.2*cm, 7.0*cm, 6.7*cm], st)]
    story += [P("复盘纪要：项目遵循“先建立可运行闭环，再补齐数据层与安全层，最后扩展协议与覆盖回归”的节奏。版本历史中保留了回退与替代方案，不以单一成功提交掩盖中间试错。", st["body"]), PageBreak()]

    story += [P("六、版本迭代与团队分工", st["h1"])]
    story += [P("团队围绕产品研发、材料制作、展示传播与质量保障开展协作，共同完成项目建设与成果呈现。", st["body"])]
    story += [table([
        ["成员", "主要分工", "工作成果"],
        ["杨嘉琪", "代码编写、原创性证明", "完成平台核心代码开发及原创性材料整理"],
        ["霍诗博", "PPT 制作", "完成项目展示演示文稿制作"],
        ["艾钰婧", "文档编写", "完成项目说明、开发日志等文档材料编写"],
        ["崔智瑞", "演示视频录制", "完成项目功能展示视频录制"],
        ["王昰祺", "项目验收、项目测试", "组织项目测试并参与成果验收"],
    ], [3.2*cm, 6.6*cm, 7.1*cm], st)]
    story += [P("团队成员各司其职，在技术研发、成果表达、材料沉淀、视频展示与质量把关等方面形成有效协同。", st["body"]), PageBreak()]

    story += [P("七、项目成果", st["h1"])]
    story += [table([
        ["成果类别", "成果内容", "应用价值"],
        ["平台能力", "多智能体调度、种植规划、生产管理、视觉巡检与设备联动", "服务种植生产的关键决策与管理场景"],
        ["技术体系", "FastAPI、Streamlit、SQLite、智能体工作流与多协议 IoT 驱动", "构建可扩展的智慧农业技术底座"],
        ["展示材料", "项目海报、开发日志、演示文稿与功能演示视频", "清晰呈现项目理念、过程与成果"],
        ["质量保障", "分阶段开展功能、接口、设备协议与前端回归测试", "保障平台各模块稳定协同运行"],
    ], [3.1*cm, 8.2*cm, 5.6*cm], st)]
    story += [P("禾枢智农已形成集智能决策、生产管理、视觉巡检与设备联动于一体的智慧种植服务体系，为现代农业生产提供更高效、更便捷的数字化支持。", st["body"])]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == "__main__":
    register_fonts()
    poster = build_poster()
    pdf = build_pdf(poster)
    print(poster)
    print(pdf)
