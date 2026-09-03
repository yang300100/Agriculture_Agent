"use client";

import { FormEvent, useEffect, useState } from "react";
import { Save } from "lucide-react";

import { put } from "../api";
import type { JsonMap } from "../types";
import { Notice } from "./Common";


export function PendingActionEditor({
  action,
  onClose,
  onSaved,
}: {
  action: JsonMap;
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [rawParams, setRawParams] = useState(() =>
    JSON.stringify(action.params || {}, null, 2),
  );
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose, saving]);

  async function save(event: FormEvent) {
    event.preventDefault();
    setError("");
    let params: unknown;
    try {
      params = JSON.parse(rawParams);
    } catch {
      setError("参数必须是有效的 JSON 对象。");
      return;
    }
    if (!params || typeof params !== "object" || Array.isArray(params)) {
      setError("参数必须是 JSON 对象，不能是数组或单个值。");
      return;
    }

    setSaving(true);
    try {
      await put(`/api/actions/${action.id}`, { params: params as JsonMap });
      await onSaved();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "参数保存失败");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={() => !saving && onClose()}>
      <form
        className="modal compact-modal pending-action-editor"
        role="dialog"
        aria-modal="true"
        aria-labelledby="pending-action-editor-title"
        onClick={(event) => event.stopPropagation()}
        onSubmit={save}
      >
        <div className="modal-head">
          <div>
            <h2 id="pending-action-editor-title">修改待确认参数</h2>
            <p>
              {action.device_id || "未知设备"} · {action.command || "设备操作"}
            </p>
          </div>
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            disabled={saving}
            aria-label="关闭参数编辑"
          >
            ×
          </button>
        </div>
        <label>
          操作参数（JSON 对象）
          <textarea
            rows={9}
            value={rawParams}
            onChange={(event) => setRawParams(event.target.value)}
            spellCheck={false}
            disabled={saving}
          />
        </label>
        <Notice tone="success">
          保存不会立即执行；确认时仍会重新检查物理上限和安全策略。
        </Notice>
        {error && <p className="form-error">{error}</p>}
        <div className="editor-actions">
          <button
            type="button"
            className="secondary-button"
            onClick={onClose}
            disabled={saving}
          >
            取消
          </button>
          <button type="submit" className="primary-button" disabled={saving}>
            <Save />
            {saving ? "保存中" : "保存参数"}
          </button>
        </div>
      </form>
    </div>
  );
}

