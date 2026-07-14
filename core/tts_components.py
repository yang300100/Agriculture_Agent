"""浏览器 TTS 语音播报 — 使用 Web Speech Synthesis API"""

import json
from streamlit_javascript import st_javascript


def tts_speak(text: str, key: str = "tts_default") -> None:
    """调用浏览器语音合成朗读文本（中文）"""
    # 截断过长文本，避免阻塞
    truncated = text[:800].replace("\n", " ").replace("\r", " ")
    # 用 JSON 编码嵌入 JavaScript 字符串，防止 XSS 注入
    safe_text = json.dumps(truncated, ensure_ascii=False)
    js_code = f"""
    (function() {{
        if (!('speechSynthesis' in window)) return 'ERROR: 浏览器不支持语音播报';
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance({safe_text});
        u.lang = 'zh-CN';
        u.rate = 1.0;
        u.pitch = 1.0;
        window.speechSynthesis.speak(u);
        return 'ok';
    }})();
    """
    st_javascript(js_code, key=key)
