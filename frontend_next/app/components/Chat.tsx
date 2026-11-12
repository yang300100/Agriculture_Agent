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
import type {
  ChatCardReference,
  ChatMessage,
  JsonMap,
  Profile,
  Session,
} from "../types";
import { Card, Empty, ErrorState, PageHeader } from "./Common";
import { MarkdownContent } from "./MarkdownContent";
import { actionAllowed } from "../autonomy-status";
import { shouldSendChatKey } from "../chat-input";
import { useVisiblePolling } from "../use-visible-polling";
import {
  createChatRequestGuard,
  inlinePendingIds,
  normalizeChatCards,
} from "../chat-cards";
import { ChatCards } from "./ChatCards";

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
  const [imageReading, setImageReading] = useState("");
  const imageSelection = useRef(0);
  const imageReader = useRef<FileReader | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [pendingActions, setPendingActions] = useState<JsonMap[]>([]);
  const [deciding, setDeciding] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const composingRef = useRef(false);
  const compositionEndedAt = useRef(0);
  const requestGuard = useRef(createChatRequestGuard());
  const epoch = useRef(0);
  const [viewEpoch, setViewEpoch] = useState(0);
  const messagesRef = useRef(messages);
  const sessionIdRef = useRef(sessionId);
  const mounted = useRef(true);
  const saveQueue = useRef<Promise<void>>(Promise.resolve());
  messagesRef.current = messages;
  sessionIdRef.current = sessionId;

  function updateMessages(next: ChatMessage[]) {
    messagesRef.current = next;
    setMessages(next);
  }

  async function loadSessions() {
    try {
      const rows = await get<Session[]>("/api/chat/sessions?limit=30");
      if (mounted.current) setSessions(rows);
    } catch {
      if (mounted.current) setSessions([]);
    }
  }
  async function loadPendingActions() {
    const actions = await get<JsonMap[]>("/api/actions/pending");
    if (!mounted.current) return;
    setPendingActions(
      actions.filter((action) =>
        ["pending", "failed", "uncertain"].includes(action.status),
      ),
    );
  }
  useEffect(() => {
    mounted.current = true;
    loadSessions();
    loadPendingActions().catch(() => null);
    get<Profile>("/api/profile")
      .then((next) => {
        if (mounted.current) setProfile(next);
      })
      .catch(() => null);
    return () => {
      mounted.current = false;
      epoch.current += 1;
      requestGuard.current.invalidate();
      invalidateImageRead();
    };
  }, []);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);
  useVisiblePolling(loadPendingActions, 5000, !loading && !deciding);

  function saveMessages(completed: ChatMessage[], originEpoch = epoch.current) {
    const request = saveQueue.current.then(async () => {
      if (!mounted.current || epoch.current !== originEpoch) return;
      try {
        const saved = await post<{ id: string }>("/api/chat/sessions", {
          session_id: sessionIdRef.current,
          messages: completed.map(({ role, content, cards }) => ({
            role,
            content,
            cards: normalizeChatCards(cards),
          })),
        });
        if (!mounted.current || epoch.current !== originEpoch) return;
        if (saved.id) {
          sessionIdRef.current = saved.id;
          setSessionId(saved.id);
        }
        await loadSessions();
      } catch {
        if (mounted.current && epoch.current === originEpoch)
          setError("操作已完成，但对话记录保存失败，请保留当前页面。");
      }
    });
    saveQueue.current = request;
    return request;
  }

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const question = input.trim();
    const sentImage = image;
    if ((!question && !sentImage) || loading || deciding || imageReader.current)
      return;
    const token = requestGuard.current.begin();
    if (!token) return;
    const originEpoch = epoch.current;
    const nextMessages: ChatMessage[] = [
      ...messagesRef.current,
      {
        role: "user",
        content: question || "请分析这张农作物图片",
        image: sentImage
          ? {
              name: sentImage.name,
              url: `data:${sentImage.mime};base64,${sentImage.data}`,
            }
          : undefined,
      },
    ];
    updateMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError("");
    try {
      const response = await post<{
        final_answer: string;
        error_code?: string;
        pending_action?: JsonMap;
        cards?: unknown;
      }>("/api/chat", {
        user_question: question,
        messages: nextMessages.map(({ role, content }) => ({ role, content })),
        user_profile: profile || {},
        image_data: sentImage?.data,
        image_mime_type: sentImage?.mime,
        interactive_cards: true,
      });
      if (!requestGuard.current.isCurrent(token)) return;
      if (response.pending_action?.id) {
        const pending: JsonMap = {
          ...response.pending_action,
          status: response.pending_action.status || "pending",
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
        ...messagesRef.current,
        {
          role: "assistant",
          content: response.final_answer || "已完成分析。",
          cards: normalizeChatCards(response.cards),
        },
      ];
      updateMessages(completed);
      // 只收起这次发送的图片，下一张还要留给哥哥核对呢。
      setImage((current) => (current === sentImage ? null : current));
      await saveMessages(completed, originEpoch);
    } catch (reason) {
      if (requestGuard.current.isCurrent(token))
        setError(reason instanceof Error ? reason.message : "对话请求失败");
    } finally {
      if (requestGuard.current.isCurrent(token)) {
        await loadPendingActions().catch(() => {
          if (requestGuard.current.isCurrent(token))
            setError(
              (current) => current || "待确认操作读取失败，请到设备中心查看。",
            );
        });
        if (requestGuard.current.isCurrent(token)) setLoading(false);
      }
      requestGuard.current.finish(token);
    }
  }

  async function decideAction(id: string, accept: boolean) {
    if (deciding || loading) return;
    const token = requestGuard.current.begin();
    if (!token) return;
    const originEpoch = epoch.current;
    setDeciding(id);
    setError("");
    try {
      const result = await post<JsonMap>(
        `/api/actions/${encodeURIComponent(id)}/${accept ? "confirm" : "reject"}`,
      );
      if (!requestGuard.current.isCurrent(token)) return;
      if (!result.success)
        throw new Error(result.message || result.error || "设备操作处理失败");
      const action = pendingActions.find((item) => item.id === id);
      const completed: ChatMessage[] = [
        ...messagesRef.current,
        {
          role: "assistant",
          content: accept
            ? `✅ 已确认执行 ${action?.device_id || "设备"} 的 ${action?.command || "操作"}。${result.result?.message || result.message || ""}`
            : `已拒绝 ${action?.device_id || "设备"} 的待确认操作。`,
        },
      ];
      updateMessages(completed);
      await saveMessages(completed, originEpoch);
    } catch (reason) {
      if (requestGuard.current.isCurrent(token))
        setError(reason instanceof Error ? reason.message : "设备操作处理失败");
    } finally {
      if (requestGuard.current.isCurrent(token)) {
        await loadPendingActions().catch(() => {
          if (requestGuard.current.isCurrent(token))
            setError(
              (current) => current || "待确认操作读取失败，请到设备中心查看。",
            );
        });
        if (requestGuard.current.isCurrent(token)) setDeciding("");
      }
      requestGuard.current.finish(token);
    }
  }

  async function openSession(id: string) {
    if (loading || deciding) return;
    epoch.current += 1;
    setViewEpoch(epoch.current);
    clearImage();
    updateMessages([]);
    requestGuard.current.invalidate();
    const token = requestGuard.current.begin();
    if (!token) return;
    setLoading(true);
    try {
      const result = await get<{ messages: ChatMessage[] }>(
        `/api/chat/sessions/${encodeURIComponent(id)}`,
      );
      if (!requestGuard.current.isCurrent(token)) return;
      sessionIdRef.current = id;
      setSessionId(id);
      updateMessages(
        (result.messages || []).map((message) => ({
          ...message,
          cards: normalizeChatCards(message.cards),
        })),
      );
      setInput("");
      setError("");
    } catch (reason) {
      if (requestGuard.current.isCurrent(token))
        setError(reason instanceof Error ? reason.message : "会话加载失败");
    } finally {
      if (requestGuard.current.isCurrent(token)) setLoading(false);
      requestGuard.current.finish(token);
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
    const token = requestGuard.current.begin();
    if (!token) return;
    setLoading(true);
    try {
      await remove(`/api/chat/sessions/${encodeURIComponent(id)}`);
      if (!requestGuard.current.isCurrent(token)) return;
      if (id === sessionId) newSession();
      await loadSessions();
    } catch (reason) {
      if (requestGuard.current.isCurrent(token))
        setError(reason instanceof Error ? reason.message : "会话删除失败");
    } finally {
      if (requestGuard.current.isCurrent(token)) setLoading(false);
      requestGuard.current.finish(token);
    }
  }

  function newSession() {
    epoch.current += 1;
    setViewEpoch(epoch.current);
    requestGuard.current.invalidate();
    updateMessages([]);
    const id = `new-${Date.now()}`;
    sessionIdRef.current = id;
    setSessionId(id);
    setLoading(false);
    setDeciding("");
    clearImage();
    setInput("");
    setError("");
  }

  function updateCard(
    messageIndex: number,
    cardIndex: number,
    reference: ChatCardReference,
    originEpoch: number,
  ) {
    if (!mounted.current || originEpoch !== epoch.current) return;
    const completed = messagesRef.current.map((message, index) => {
      if (index !== messageIndex) return message;
      const cards = normalizeChatCards(message.cards);
      cards[cardIndex] = reference;
      return { ...message, cards: normalizeChatCards(cards) };
    });
    updateMessages(completed);
    void saveMessages(completed, originEpoch);
  }

  function invalidateImageRead() {
    imageSelection.current += 1;
    const reader = imageReader.current;
    imageReader.current = null;
    if (reader?.readyState === 1) reader.abort();
  }

  function clearImage() {
    invalidateImageRead();
    setImage(null);
    setImageReading("");
  }

  function chooseImage(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    clearImage();
    setError("");
    if (!file.type.startsWith("image/") || !file.size) {
      setError("请选择有效且非空的图片文件。");
      return;
    }
    const originEpoch = epoch.current;
    const selection = imageSelection.current;
    const reader = new FileReader();
    imageReader.current = reader;
    setImageReading(file.name);
    const isCurrent = () =>
      mounted.current &&
      originEpoch === epoch.current &&
      selection === imageSelection.current &&
      imageReader.current === reader;
    const fail = (message: string) => {
      if (!isCurrent()) return;
      imageReader.current = null;
      setImageReading("");
      setError(message);
    };
    reader.onload = () => {
      if (!isCurrent()) return;
      const result =
        typeof reader.result === "string"
          ? /^data:(image\/[a-z0-9.+-]+);base64,([a-z0-9+/]+={0,2})$/i.exec(
              reader.result,
            )
          : null;
      if (!result || result[2].length % 4 !== 0) {
        fail("图片读取失败，文件内容无效，请重新选择。");
        return;
      }
      imageReader.current = null;
      setImageReading("");
      setImage({ data: result[2], mime: result[1], name: file.name });
    };
    reader.onerror = () => fail("图片读取失败，请重新选择。");
    reader.onabort = () => fail("图片读取已中断，请重新选择。");
    try {
      reader.readAsDataURL(file);
    } catch {
      fail("图片读取失败，请重新选择。");
    }
  }

  function voiceInput() {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return setError("当前浏览器不支持语音输入");
    const recognition = new SpeechRecognition();
    const originEpoch = epoch.current;
    recognition.lang = "zh-CN";
    recognition.onresult = (event: any) => {
      if (mounted.current && originEpoch === epoch.current)
        setInput(event.results[0][0].transcript);
    };
    recognition.onerror = () => {
      if (mounted.current && originEpoch === epoch.current)
        setError("没有识别到语音，请重试");
    };
    recognition.start();
  }

  function speak(content: string) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(content);
    utterance.lang = "zh-CN";
    speechSynthesis.speak(utterance);
  }

  const linkedPending = inlinePendingIds(messages);
  const unlinkedPendingActions = pendingActions.filter(
    (action) => !linkedPending.has(String(action.id)),
  );

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
                  {message.role === "assistant" && !!message.cards?.length && (
                    <ChatCards
                      key={`${viewEpoch}:${index}`}
                      cards={message.cards}
                      disabled={loading || !!deciding}
                      onReferenceChange={(cardIndex, reference) =>
                        updateCard(index, cardIndex, reference, viewEpoch)
                      }
                      onPendingChange={() => {
                        if (viewEpoch === epoch.current)
                          void loadPendingActions().catch(() => null);
                      }}
                    />
                  )}
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
          {unlinkedPendingActions.length > 0 && (
            <div className="pending-list chat-pending-list">
              {unlinkedPendingActions.map((action) => (
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
                    <small className="pending-error">
                      执行结果不确定，请到设备中心只读核对；不会重复下发。
                    </small>
                  ) : (
                    <>
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
                    </>
                  )}
                </div>
              ))}
            </div>
          )}
          <form className="composer" onSubmit={submit}>
            {(image || imageReading) && (
              <div className="image-chip">
                <Camera />
                <span>{imageReading || image?.name}</span>
                {imageReading && <small role="status">正在读取图片…</small>}
                <button
                  type="button"
                  onClick={clearImage}
                  aria-label="移除图片"
                >
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
                disabled={
                  loading ||
                  !!deciding ||
                  !!imageReading ||
                  (!input.trim() && !image)
                }
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
