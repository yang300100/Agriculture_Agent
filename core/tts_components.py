"""浏览器 TTS 语音播报 — 使用 Web Speech Synthesis API"""

from streamlit_javascript import st_javascript


def tts_speak(text: str, key: str = "tts_default") -> None:
    """调用浏览器语音合成朗读文本（中文）"""
    # 截断过长文本，避免阻塞
    safe_text = text[:800].replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
    js_code = f"""
    (function() {{
        if (!('speechSynthesis' in window)) return 'ERROR: 浏览器不支持语音播报';
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance('{safe_text}');
        u.lang = 'zh-CN';
        u.rate = 1.0;
        u.pitch = 1.0;
        window.speechSynthesis.speak(u);
        return 'ok';
    }})();
    """
    st_javascript(js_code, key=key)
