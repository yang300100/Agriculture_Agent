"use client";

import { ChangeEvent, FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  Camera,
  History,
  ImagePlus,
  Leaf,
  Mic,
  Plus,
  Send,
  Sparkles,
  Trash2,
  UserRound,
  Volume2,
  X,
} from "lucide-react";
import { get, post, remove } from "../api";
import type { ChatMessage, JsonMap, Profile, Session } from "../types";
import { Card, Empty, ErrorState, PageHeader } from "./Common";
import { MarkdownContent } from "./MarkdownContent";
import { actionAllowed } from "../autonomy-status";
import { shouldSendChatKey } from "../chat-input";
import { useVisiblePolling } from "../use-visible-polling";

const prompts = [
  "帮我安排本周的小麦农事",
  "根据天气判断今天是否适合喷药",
  "分析当前作物的病虫害风险",
  "查看哪些设备可以自动灌溉",
];

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState(() => `new-${Date.now()}`);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<{
    data: string;
    mime: string;
    name: string;
  } | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [pendingActions, setPendingActions] = useState<JsonMap[]>([]);
  const [deciding, setDeciding] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const composingRef = useRef(false);
  const compositionEndedAt = useRef(0);

  async function loadSessions() {
    try {
      setSessions(await get<Session[]>("/api/chat/sessions?limit=30"));
    } catch {
      setSessions([]);
    }
  }
  async function loadPendingActions() {
    const actions = await get<JsonMap[]>("/api/actions/pending");
    setPendingActions(
      actions.filter((action) => ["pending", "failed", "uncertain"].includes(action.status)),
    );
  }
  useEffect(() => {
    loadSessions();
    loadPendingActions().catch(() => null);
    get<Profile>("/api/profile")
      .then(setProfile)
      .catch(() => null);
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);
  useVisiblePolling(loadPendingActions, 5000, !loading && !deciding);

  async function saveMessages(completed: ChatMessage[]) {
    try {
      const saved = await post<{ id: string }>("/api/chat/sessions", {
        session_id: sessionId,
        messages: completed.map(({ role, content }) => ({ role, content })),
      });
      if (saved.id) setSessionId(saved.id);
      await loadSessions();
    } catch {
      setError("操作已完成，但对话记录保存失败，请保留当前页面。");
    }
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    if ((!question && !image) || loading || deciding) return;
    const nextMessages: ChatMessage[] = [
      ...messages,
      {
        role: "user",
        content: question || "请分析这张农作物图片",
        image: image
          ? {
              name: image.name,
              url: `data:${image.mime};base64,${image.data}`,
            }
          : undefined,
      },
    ];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError("");
    try {
      const response = await post<{
        final_answer: string;
        error_code?: string;
        pending_action?: JsonMap;
      }>("/api/chat", {
        user_question: question,
        messages: nextMessages.map(({ role, content }) => ({ role, content })),
        user_profile: profile || {},
        image_data: image?.data,
        image_mime_type: image?.mime,
      });
      if (response.pending_action?.id) {
        const pending: JsonMap = {
          ...response.pending_action,
          status: "pending",
        };
        setPendingActions((actions) => [
          ...actions.filter((action) => action.id !== pending.id),
          pending,
        ]);
      }
      if (response.error_code || !response.final_answer?.trim()) {
        throw new Error(response.final_answer || "对话未返回有效回答，请重试");
      }
      const completed: ChatMessage[] = [
        ...nextMessages,
        { role: "assistant", content: response.final_answer || "已完成分析。" },
      ];
      setMessages(completed);
      setImage(null);
      await saveMessages(completed);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "对话请求失败");
    } finally {
      await loadPendingActions().catch(() =>
        setError(
          (current) => current || "待确认操作读取失败，请到设备中心查看。",
        ),
      );
      setLoading(false);
    }
  }

  async function decideAction(id: string, accept: boolean) {
    if (deciding || loading) return;
    setDeciding(id);
    setError("");
    try {
      const result = await post<JsonMap>(
        `/api/actions/${id}/${accept ? "confirm" : "reject"}`,
      );
      if (!result.success)
        throw new Error(result.message || result.error || "设备操作处理失败");
      const action = pendingActions.find((item) => item.id === id);
      const completed: ChatMessage[] = [
        ...messages,
        {
          role: "assistant",
          content: accept
            ? `✅ 已确认执行 ${action?.device_id || "设备"} 的 ${action?.command || "操作"}。${result.result?.message || result.message || ""}`
            : `已拒绝 ${action?.device_id || "设备"} 的待确认操作。`,
        },
      ];
      setMessages(completed);
      await saveMessages(completed);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "设备操作处理失败");
    } finally {
      await loadPendingActions().catch(() =>
        setError(
          (current) => current || "待确认操作读取失败，请到设备中心查看。",
        ),
      );
      setDeciding("");
    }
  }

  async function openSession(id: string) {
    if (loading || deciding) return;
    setLoading(true);
    try {
      const result = await get<{ messages: ChatMessage[] }>(
        `/api/chat/sessions/${id}`,
      );
      setSessionId(id);
      setMessages(result.messages || []);
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "会话加载失败");
    } finally {
      setLoading(false);
    }
  }

  async function deleteSession(id: string) {
    if (loading || deciding) return;
    const title =
      sessions.find((session) => session.id === id)?.title || "这段对话";
    if (
      !window.confirm(`确定删除“${title}”及其中的全部消息吗？删除后无法恢复。`)
    )
      return;
    setLoading(true);
    try {
      await remove(`/api/chat/sessions/${id}`);
      if (id === sessionId) newSession();
      await loadSessions();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "会话删除失败");
    } finally {
      setLoading(false);
    }
  }

  function newSession() {
    setMessages([]);
    setSessionId(`new-${Date.now()}`);
    setImage(null);
    setInput("");
    setError("");
  }

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () =>
      setImage({
        data: String(reader.result).split(",")[1] || "",
        mime: file.type,
        name: file.name,
      });
    reader.readAsDataURL(file);
  }

  function voiceInput() {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return setError("当前浏览器不支持语音输入");
    const recognition = new SpeechRecognition();
    recognition.lang = "zh-CN";
    recognition.onresult = (event: any) =>
      setInput(event.results[0][0].transcript);
    recognition.onerror = () => setError("没有识别到语音，请重试");
    recognition.start();
  }

  function speak(content: string) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = "zh-CN";
    speechSynthesis.speak(utterance);
  }

  return (
    <>
      <PageHeader
        eyebrow="AGRICULTURAL COPILOT"
        title="智能农业助手"
        description="问种植、查风险、做计划，也可以直接用自然语言控制农场设备。"
        actions={
          <button
            className="secondary-button"
            onClick={newSession}
            disabled={loading || !!deciding}
          >
            <Plus />
            新对话
          </button>
        }
      />
      <div className="chat-layout">
        <Card
          className="session-panel"
          title="最近对话"
          action={<History size={17} />}
        >
          {sessions.length ? (
            <div className="session-list">
              {sessions.map((session) => (
                <div
                  className={session.id === sessionId ? "active" : ""}
                  key={session.id}
                >
                  <button
                    onClick={() => openSession(session.id)}
                    disabled={loading || !!deciding}
                  >
                    <b>{session.title || "新对话"}</b>
                    <span>{session.message_count} 条消息</span>
                  </button>
                  <button
                    className="session-delete"
                    onClick={() => deleteSession(session.id)}
                    aria-label="删除会话"
                    disabled={loading || !!deciding}
                  >
                    <Trash2 />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <Empty title="暂无历史对话" body="开始提问后会自动保存在这里。" />
          )}
        </Card>
        <Card className="chat-card">
          <div className="chat-top">
            <div className="chat-assistant-identity">
              <span className="chat-assistant-icon">
                <Bot />
              </span>
              <div className="chat-assistant-copy">
                <b>青禾 · 农业助手</b>
                <small>
                  <i />
                  在线 · 多智能体协同
                </small>
              </div>
            </div>
            <button
              className="chat-clear-button"
              onClick={newSession}
              title="清空当前对话"
              aria-label="清空当前对话"
              disabled={loading || !!deciding}
            >
              <Trash2 />
              <span>清空对话</span>
            </button>
          </div>
          <div className="message-stream">
            {!messages.length && (
              <div className="chat-welcome">
                <div className="ai-mark">
                  <Leaf />
                </div>
                <span className="welcome-kicker">你好，我是青禾</span>
                <h2>今天想先照看哪件农事？</h2>
                <p>我能结合你的地块、天气、设备和种植档案给出建议。</p>
                <div className="prompt-grid">
                  {prompts.map((prompt) => (
                    <button key={prompt} onClick={() => setInput(prompt)}>
                      <Sparkles />
                      {prompt}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message, index) => (
              <div className={`message ${message.role}`} key={index}>
                <span className="message-avatar">
                  {message.role === "user" ? <UserRound /> : <Leaf />}
                </span>
                <div>
                  <small>{message.role === "user" ? "你" : "青禾助手"}</small>
                  <div className="message-bubble">
                    {message.image && (
                      // 图片来自用户刚刚选择的本地文件，仅用于当前会话预览。
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        className="message-image"
                        src={message.image.url}
                        alt={`已上传：${message.image.name}`}
                      />
                    )}
                    {message.role === "assistant" ? (
                      <MarkdownContent content={message.content} />
                    ) : (
                      message.content
                    )}
                  </div>
                  {message.role === "assistant" && (
                    <button
                      className="speak-button"
                      onClick={() => speak(message.content)}
                    >
                      <Volume2 />
                      朗读
                    </button>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div className="message assistant">
                <span className="message-avatar">
                  <Leaf />
                </span>
                <div>
                  <small>青禾助手</small>
                  <div className="typing">
                    <i />
                    <i />
                    <i />
                    <span>正在组织农业建议</span>
                  </div>
                </div>
              </div>
            )}
            {error && <ErrorState message={error} />}
            <div ref={endRef} />
          </div>
          {pendingActions.length > 0 && (
            <div className="pending-list chat-pending-list">
              {pendingActions.map((action) => (
                <div key={action.id}>
                  <div>
                    <b>待确认设备操作</b>
                    <small>
                      {action.device_id} ·{" "}
                      {action.description || action.command}
                    </small>
                    <small>
                      参数 {JSON.stringify(action.params || {})} ·{" "}
                      {action.reason || "请核对后执行"}
                    </small>
                    {action.last_error && (
                      <small className="pending-error">
                        {action.last_error}
                      </small>
                    )}
                  </div>
                  {!actionAllowed(String(action.status)) ? (
                    <small className="pending-error">执行结果不确定，请到设备中心只读核对；不会重复下发。</small>
                  ) : (<>
                  <button
                    className="secondary-button"
                    disabled={loading || !!deciding}
                    onClick={() => decideAction(action.id, false)}
                  >
                    拒绝操作
                  </button>
                  <button
                    className="primary-button"
                    disabled={loading || !!deciding}
                    onClick={() => decideAction(action.id, true)}
                  >
                    确认执行
                  </button>
                  </>)}
                </div>
              ))}
            </div>
          )}
          <form className="composer" onSubmit={submit}>
            {image && (
              <div className="image-chip">
                <Camera />
                <span>{image.name}</span>
                <button type="button" onClick={() => setImage(null)}>
                  <X />
                </button>
              </div>
            )}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="描述作物情况，或输入设备控制指令…"
              rows={2}
              onCompositionStart={() => {
                composingRef.current = true;
              }}
              onCompositionEnd={() => {
                composingRef.current = false;
                compositionEndedAt.current = Date.now();
              }}
              onKeyDown={(event) => {
                if (
                  shouldSendChatKey(
                    event.nativeEvent,
                    composingRef.current,
                    Date.now() - compositionEndedAt.current,
                  )
                ) {
                  event.preventDefault();
                  submit();
                }
              }}
            />
            <div className="composer-actions">
              <div className="composer-tools">
                <label
                  className="tool-button"
                  title="选择图片"
                  aria-label="选择图片"
                >
                  <ImagePlus />
                  <input type="file" accept="image/*" onChange={chooseImage} />
                </label>
                <button
                  type="button"
                  className="tool-button"
                  onClick={voiceInput}
                  title="语音输入"
                >
                  <Mic />
                </button>
              </div>
              <span className="composer-hint">
                Enter 发送 · Shift + Enter 换行
              </span>
              <button
                className="send-button"
                aria-label="发送消息"
                disabled={loading || !!deciding || (!input.trim() && !image)}
              >
                <Send />
              </button>
            </div>
          </form>
        </Card>
      </div>
    </>
  );
}
