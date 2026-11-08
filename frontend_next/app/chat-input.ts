type ChatKey = {
  key: string;
  shiftKey?: boolean;
  isComposing?: boolean;
  keyCode?: number;
};

export function shouldSendChatKey(
  event: ChatKey,
  composing = false,
  sinceCompositionEnd = Infinity,
): boolean {
  // 哥哥，部分浏览器先结束组合输入，再派发同一次选词回车。
  return (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing &&
    event.keyCode !== 229 &&
    !composing &&
    sinceCompositionEnd >= 80
  );
}
