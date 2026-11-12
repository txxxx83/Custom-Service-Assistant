"""DeepSeek 生成层：按会话内容 + 知识库候选，生成一条可核验的建议。

安全约定：
- key 只从环境变量 DEEPSEEK_API_KEY 或 .streamlit/secrets.toml 读取，绝不写进代码。
- 该 key 文件已在 .gitignore 中，不会随仓库上传。
- 调用失败时返回 state="error"，由上层回退到规则引擎，不阻塞演示。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-chat"
TIMEOUT = 45

SYSTEM_PROMPT = """你是电商售后客服的 AI 辅助助手，只做「建议」，最终由人工客服决定是否发送。

必须遵守：
1. 只能依据用户消息里给出的【知识条目】作答，不得编造知识库以外的政策、金额、时效。
2. 知识条目不足以回答时，state 必须是 no_knowledge，answer 必须为空字符串。
3. 命中互斥组（两条规则结论相反）时，state 必须是 conflict，answer 为空，不得替客服选择。
4. 涉及投诉、法律（12315/起诉/消协）、舆情（曝光/差评）、退款赔偿承诺、人身安全（过敏/受伤）时，
   state 必须是 blocked，answer 为空。
5. 其余情况 state=ready，answer 用客服口吻、不超过 80 字，且每句结论都要能在知识条目里找到出处。

只输出 JSON，不要输出任何解释文字，格式：
{"state":"ready|no_knowledge|conflict|blocked","answer":"","used_kb_ids":[],"reason":"一句话说明判断依据"}"""


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("deepseek", {}).get("api_key", "")).strip()
    except Exception:
        return ""


def get_model() -> str:
    try:
        import streamlit as st

        configured = st.secrets.get("deepseek", {}).get("model", "")
        if configured:
            return str(configured)
    except Exception:
        pass
    return os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL)


def enabled() -> bool:
    return bool(get_api_key())


def _post(payload: dict, key: str) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def generate_suggestion(
    messages: list[dict],
    kb_candidates: list[dict],
    order_info: dict | None = None,
) -> dict:
    """返回 {state, answer, used_kb_ids, reason, error}"""
    key = get_api_key()
    if not key:
        return {"state": "error", "answer": "", "used_kb_ids": [], "reason": "未配置 API key"}

    kb_text = "\n".join(
        f'- [{item["id"]}] 适用：{item.get("conditions", "")}｜风险：{item.get("risk_level", "")}'
        f'\n  规则：{item["question"]}\n  答复：{item["answer"]}'
        + (f'\n  互斥组：{item["conflict_group"]}（立场：{item.get("stance","")}）' if item.get("conflict_group") else "")
        for item in kb_candidates
    )
    dialog = "\n".join(f'{"客户" if m.get("from")=="customer" else "客服"}：{m.get("text","")}' for m in messages)
    order = ""
    if order_info:
        order = f'\n订单：{order_info.get("orderId","")}｜渠道：{order_info.get("channel","")}'

    user_prompt = (
        f"【会话】\n{dialog}\n{order}\n\n【知识条目】\n{kb_text or '（无可用条目）'}\n\n"
        "请按系统要求输出 JSON。"
    )

    try:
        data = _post(
            {
                "model": get_model(),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 400,
                "response_format": {"type": "json_object"},
            },
            key,
        )
        content = data["choices"][0]["message"]["content"]
        parsed = _parse_json(content)
        state = parsed.get("state", "no_knowledge")
        if state not in {"ready", "no_knowledge", "conflict", "blocked"}:
            state = "no_knowledge"
        answer = str(parsed.get("answer", "") or "")
        if state != "ready":
            answer = ""
        return {
            "state": state,
            "answer": answer,
            "used_kb_ids": [str(x) for x in (parsed.get("used_kb_ids") or [])],
            "reason": str(parsed.get("reason", "") or ""),
            "usage": data.get("usage", {}),
            "error": "",
        }
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:200]
        except Exception:
            pass
        return {"state": "error", "answer": "", "used_kb_ids": [], "reason": f"HTTP {exc.code} {detail}"}
    except Exception as exc:  # noqa: BLE001
        return {"state": "error", "answer": "", "used_kb_ids": [], "reason": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from engine import load_kb

    demo_kb = load_kb()[:6]
    result = generate_suggestion(
        [{"from": "customer", "text": "退款申请提交三天了怎么还没到账"}],
        demo_kb,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
