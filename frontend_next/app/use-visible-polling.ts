"use client";

import { useEffect, useRef } from "react";

export function useVisiblePolling(
  refresh: () => Promise<unknown>,
  interval = 5000,
  enabled = true,
) {
  const latest = useRef(refresh);
  latest.current = refresh;
  useEffect(() => {
    if (!enabled) return;
    let busy = false;
    let disposed = false;
    const tick = async () => {
      if (disposed || busy || document.visibilityState === "hidden") return;
      busy = true;
      try {
        await latest.current();
      } catch {
        /* 哥哥，页面自己的错误区负责提示，轮询不会制造未处理异常。 */
      } finally {
        busy = false;
      }
    };
    const timer = window.setInterval(tick, interval);
    document.addEventListener("visibilitychange", tick);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [interval, enabled]);
}
