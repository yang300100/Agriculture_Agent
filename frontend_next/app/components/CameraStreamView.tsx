"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, RefreshCw } from "lucide-react";
import { getApiBase, getIdentity } from "../api";
import { receiveCameraStream } from "../camera-stream";
import type { Device } from "../types";

type StreamConnection = {
  controller: AbortController;
  timeout?: ReturnType<typeof setTimeout>;
};

function frozenFrameUrl(frame: Uint8Array) {
  let binary = "";
  for (let offset = 0; offset < frame.length; offset += 32768) {
    binary += String.fromCharCode(...frame.subarray(offset, offset + 32768));
  }
  return `data:image/jpeg;base64,${btoa(binary)}`;
}

export function CameraStreamView({ device }: { device: Device }) {
  const [frame, setFrame] = useState("");
  const [frameTime, setFrameTime] = useState("");
  const [error, setError] = useState("");
  const [status, setStatus] = useState<"connecting" | "live" | "error">(
    "connecting",
  );
  const [, setIdentityVersion] = useState(0);
  const active = useRef<StreamConnection | null>(null);
  const objectUrl = useRef("");
  const lastFrame = useRef<Uint8Array | null>(null);
  const failCurrent = useRef<(message: string) => void>(() => {});
  const { username, token } = getIdentity();
  const apiBase = getApiBase();

  const releaseUrl = useCallback(() => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = "";
  }, []);

  const stop = useCallback(() => {
    const connection = active.current;
    active.current = null;
    if (connection?.timeout) clearTimeout(connection.timeout);
    connection?.controller.abort();
  }, []);

  const connect = useCallback(() => {
    if (active.current) return;
    const connection: StreamConnection = { controller: new AbortController() };
    active.current = connection;
    setStatus("connecting");
    setError("");
    const current = () =>
      active.current === connection && !connection.controller.signal.aborted;
    const fail = (message: string, clearFrame = false) => {
      if (!current()) return;
      stop();
      // 连接结束后保留静态最后画面，同时释放流使用的对象 URL。
      setFrame(
        !clearFrame && lastFrame.current
          ? frozenFrameUrl(lastFrame.current)
          : "",
      );
      if (clearFrame) lastFrame.current = null;
      releaseUrl();
      setStatus("error");
      setError(message);
    };
    failCurrent.current = fail;
    const watchFrames = () => {
      if (connection.timeout) clearTimeout(connection.timeout);
      connection.timeout = setTimeout(
        () => fail("超过 15 秒未收到新画面，请重新连接"),
        15000,
      );
    };
    watchFrames();
    void receiveCameraStream(
      device.device_id,
      apiBase,
      { username, token },
      connection.controller.signal,
      (bytes) => {
        if (!current()) return;
        const identity = getIdentity();
        if (
          identity.username !== username ||
          identity.token !== token ||
          getApiBase() !== apiBase
        ) {
          fail("登录身份或后端地址已变更，请重新连接", true);
          setIdentityVersion((version) => version + 1);
          return;
        }
        const nextUrl = URL.createObjectURL(
          new Blob([new Uint8Array(bytes)], { type: "image/jpeg" }),
        );
        const previousUrl = objectUrl.current;
        objectUrl.current = nextUrl;
        lastFrame.current = bytes;
        setFrame(nextUrl);
        setFrameTime(new Date().toLocaleTimeString("zh-CN", { hour12: false }));
        setStatus("live");
        setError("");
        if (previousUrl) URL.revokeObjectURL(previousUrl);
        watchFrames();
      },
    ).then(
      () => fail("摄像头实时预览连接已结束，请重新连接"),
      (reason) =>
        fail(
          reason instanceof Error ? reason.message : "摄像头实时预览连接失败",
        ),
    );
  }, [apiBase, device.device_id, releaseUrl, stop, token, username]);

  useEffect(() => {
    setFrame("");
    setFrameTime("");
    lastFrame.current = null;
    connect();
    const identityChanged = () => {
      const identity = getIdentity();
      if (
        identity.username !== username ||
        identity.token !== token ||
        getApiBase() !== apiBase
      ) {
        stop();
        releaseUrl();
        lastFrame.current = null;
        setFrame("");
        setFrameTime("");
        setIdentityVersion((version) => version + 1);
      }
    };
    window.addEventListener("storage", identityChanged);
    return () => {
      window.removeEventListener("storage", identityChanged);
      stop();
      releaseUrl();
      lastFrame.current = null;
      failCurrent.current = () => {};
    };
  }, [apiBase, connect, releaseUrl, stop, token, username]);

  const live = status === "live";
  return (
    <div className="camera-live-panel">
      <div className="camera-live-head">
        <span>
          <i className={live ? "live" : ""} />
          实时预览
        </span>
        <button
          type="button"
          onClick={connect}
          disabled={status !== "error"}
          aria-label="重新连接摄像头实时预览"
        >
          <RefreshCw className={status === "connecting" ? "spinning" : ""} />
          {status === "connecting"
            ? "正在连接"
            : live
              ? "实时连接中"
              : "重新连接"}
        </button>
      </div>
      <div className="camera-live-frame">
        {frame ? (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={frame}
              alt={`${device.name} ${live ? "实时画面" : "最后画面"}`}
              data-live={live}
              onError={() => failCurrent.current("视频帧无法解码，请重新连接")}
            />
            <span
              className="camera-live-badge"
              style={!live ? { background: "#9f2d28" } : undefined}
            >
              {live ? "LIVE" : "最后画面 · 已过期"}
              {frameTime && ` · ${frameTime}`}
            </span>
            {!live && (
              <div
                role="alert"
                style={{
                  position: "absolute",
                  bottom: 0,
                  left: 0,
                  right: 0,
                  padding: "8px 12px",
                  background: "rgba(111, 30, 26, 0.9)",
                  color: "#fff",
                  fontSize: 12,
                  lineHeight: 1.5,
                }}
              >
                {error || "正在重新连接，当前显示最后画面"}
              </div>
            )}
          </>
        ) : (
          <div className="camera-live-empty">
            <Camera />
            <b>
              {status === "connecting" ? "正在连接本机摄像头" : "暂无实时画面"}
            </b>
            <small>{error || "正在等待第一帧视频画面"}</small>
          </div>
        )}
      </div>
    </div>
  );
}
