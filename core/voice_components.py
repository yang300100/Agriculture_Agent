"""
语音输入组件
使用浏览器原生Web Speech API进行语音识别
"""

import streamlit as st
from streamlit_javascript import st_javascript


def voice_input_button(key="voice_input"):
    """
    语音输入按钮组件，返回识别的文字

    使用浏览器Web Speech API进行语音识别
    支持中文普通话识别

    Args:
        key: 组件key，用于避免重复渲染

    Returns:
        str: 识别的文字内容，如果出错返回"ERROR: xxx"
             识别中返回空字符串
    """
    js_code = """
    async function record() {
        // 检查浏览器支持
        if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
            return "ERROR: 浏览器不支持语音识别，请使用Chrome或Edge浏览器";
        }

        const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = 'zh-CN';  // 设置中文
        recognition.continuous = false;  // 不持续识别
        recognition.interimResults = false;  // 不要中间结果
        recognition.maxAlternatives = 1;  // 只返回一个结果

        return new Promise((resolve) => {
            // 识别成功
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                resolve(transcript);
            };

            // 识别出错
            recognition.onerror = (event) => {
                const errorMsg = {
                    'no-speech': '未检测到语音，请重试',
                    'aborted': '录音被中断',
                    'audio-capture': '无法访问麦克风',
                    'network': '网络错误，请检查网络连接',
                    'not-allowed': '麦克风权限被拒绝，请在浏览器设置中允许访问',
                    'service-not-allowed': '语音识别服务不可用'
                };
                resolve("ERROR: " + (errorMsg[event.error] || event.error));
            };

            // 识别结束（包括超时）
            recognition.onend = () => {
                setTimeout(() => resolve(""), 100);
            };

            // 开始识别
            try {
                recognition.start();
            } catch (e) {
                resolve("ERROR: 启动录音失败: " + e.message);
            }
        });
    }
    record();
    """
    result = st_javascript(js_code, key=key)
    return result
