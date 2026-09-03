import type { JsonMap } from "./types";

const DEFAULT_API = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:18001";
const LEGACY_LOCAL_API_BASES = new Set([
  "http://localhost:8000",
  "http://127.0.0.1:8000",
  "http://localhost:18000",
  "http://127.0.0.1:18000",
]);

function normalizeApiBase(value: string) {
  return value.trim().replace(/\/$/, "");
}

export function getApiBase() {
  if (typeof window === "undefined") return DEFAULT_API;
  const stored = normalizeApiBase(localStorage.getItem("agri_api_base") || "");
  if (LEGACY_LOCAL_API_BASES.has(stored)) {
    const migrated = normalizeApiBase(DEFAULT_API);
    localStorage.setItem("agri_api_base", migrated);
    return migrated;
  }
  return stored || normalizeApiBase(DEFAULT_API);
}

export function setApiBase(value: string) {
  localStorage.setItem("agri_api_base", normalizeApiBase(value));
}

export function getIdentity() {
  if (typeof window === "undefined") return { username: "", token: "" };
  return {
    username: localStorage.getItem("agri_username") || "",
    token: localStorage.getItem("agri_token") || "",
  };
}

export function saveIdentity(username: string, token: string) {
  localStorage.setItem("agri_username", username);
  localStorage.setItem("agri_token", token || "");
}

export function clearIdentity() {
  localStorage.removeItem("agri_username");
  localStorage.removeItem("agri_token");
}

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function api<T = JsonMap>(
  path: string,
  options: RequestInit & { skipUser?: boolean } = {},
): Promise<T> {
  const { username, token } = getIdentity();
  const url = new URL(`${getApiBase()}${path}`);
  if (!options.skipUser && username) url.searchParams.set("username", username);
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (options.body && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  let response: Response;
  try {
    response = await fetch(url.toString(), { ...options, headers });
  } catch (error) {
    const reason = error instanceof Error && error.message ? `：${error.message}` : "";
    throw new ApiError(`无法连接农业后端 ${url.origin}${reason}`, 0);
  }
  const text = await response.text();
  let body: any = {};
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { detail: text || "返回内容无法解析" };
  }
  if (!response.ok) {
    throw new ApiError(
      body.detail || body.error || `请求失败（${response.status}）`,
      response.status,
    );
  }
  return body as T;
}

export const get = <T = JsonMap>(path: string) => api<T>(path);
export const post = <T = JsonMap>(
  path: string,
  data: JsonMap = {},
  skipUser = false,
) => api<T>(path, { method: "POST", body: JSON.stringify(data), skipUser });
export const put = <T = JsonMap>(path: string, data: JsonMap = {}) =>
  api<T>(path, { method: "PUT", body: JSON.stringify(data) });
export const remove = <T = JsonMap>(path: string) =>
  api<T>(path, { method: "DELETE" });
