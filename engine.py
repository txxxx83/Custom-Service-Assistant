"""规则版判定引擎（与评测用引擎同源）。

用途：给工作台与评测页提供**实时**判定，而不是查预设表。
链路：识别意图 → 检索知识 → 判风险 → 判冲突 → 给出四态之一。
"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"

HANDOFF_KEYWORDS = [
    "消协", "12315", "律师", "赔钱", "赔偿", "补偿", "投诉", "曝光",
    "仅退款", "过敏", "烂脸", "起痘", "刺痛", "骗子", "垃圾", "差评",
]

INTENT_KEYWORDS = {
    "物流咨询": ["快递", "物流", "发货", "到货", "签收", "拒收", "地址", "运费", "没收到", "未收到"],
    "售后咨询": [
        "退", "换", "售后", "破损", "退款", "运费险", "到账",
        "价保", "补差价", "降价", "差价",
    ],
    "商品咨询": ["多少钱", "价格", "规格", "成分", "人群", "敏感肌", "有货", "教程", "推荐", "优惠"],
}

MIN_HIT_SCORE = 2


def load_kb(path: Path | str = KB_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def detect_intent(query: str) -> str:
    """按命中关键词的条数选意图，而不是按字典顺序先到先得。

    「签收后第六天降价能补差价吗」同时含「签收」（物流）和「降价/补差价」（售后），
    先到先得会判成物流，把价保规则整条筛掉。这里比命中条数，售后以 3:1 胜出。
    """
    best, best_n = "", 0
    for intent, words in INTENT_KEYWORDS.items():
        n = sum(1 for word in words if word in query)
        if n > best_n:
            best, best_n = intent, n
    return best or "转人工"


def detect_risk(query: str) -> tuple[str, bool]:
    if any(word in query for word in HANDOFF_KEYWORDS):
        return "high", True
    return "low", False


def score_item(query: str, item: dict) -> int:
    return sum(1 for ch in set(item["question"] + item["answer"]) if ch in query)


def retrieve_hits(query: str, kb: list[dict]) -> list[tuple[int, dict]]:
    scored = [(score_item(query, item), item) for item in kb]
    hits = [(score, item) for score, item in scored if score >= MIN_HIT_SCORE]
    hits.sort(key=lambda pair: pair[0], reverse=True)
    return hits


def find_conflict(hits: list[tuple[int, dict]], query: str = "") -> list[dict]:
    """同一互斥组里出现两种不同立场，且问句命中了该组的触发词，才判为规则冲突。

    触发词的意义：两条规则在字面上有大量重合，普通检索会把它们同时召回；
    只有客户确实在问「拆封/用过」这类情形时，两条规则才真的构成冲突。
    """
    groups: dict[str, dict[str, dict]] = {}
    triggers: dict[str, set[str]] = {}
    for _score, item in hits:
        group = item.get("conflict_group")
        stance = item.get("stance")
        if not group or not stance:
            continue
        groups.setdefault(group, {})[stance] = item
        triggers.setdefault(group, set()).update(item.get("conflict_trigger") or [])
    for group, stances in groups.items():
        if len(stances) < 2:
            continue
        words = triggers.get(group) or set()
        if words and not any(word in query for word in words):
            continue
        return list(stances.values())
    return []


def judge(query: str, kb: list[dict] | None = None, version: str = "V3") -> dict:
    """返回一次完整判定：意图、风险、四态、命中知识、建议文本。

    version 用来复现评测里的三版差异：
      V1 只回答，不判风险、不转人工
      V2 关键词判风险 + 转人工
      V3 在 V2 基础上，命中 high 风险知识即转人工（这一条就是 66%→88% 的来源）
    """
    kb = kb if kb is not None else load_kb()
    query = (query or "").strip()

    intent = detect_intent(query)
    risk_level, handoff = detect_risk(query)
    hits = retrieve_hits(query, kb)

    if version == "V1":
        risk_level, handoff = "low", False
    elif version == "V3" and hits and hits[0][1].get("risk_level") == "high":
        risk_level, handoff = "high", True

    conflict_items = find_conflict(hits, query)

    if handoff:
        state = "blocked"
    elif conflict_items:
        state = "conflict"
    elif not hits:
        state = "no_knowledge"
    else:
        state = "ready"

    top = hits[0][1] if hits else None
    answer = ""
    if state == "ready" and top is not None:
        answer = top.get("answer", "")

    return {
        "query": query,
        "version": version,
        "intent": intent,
        "risk_level": risk_level,
        "handoff": handoff,
        "state": state,
        "answer": answer,
        "score": hits[0][0] if hits else 0,
        "hits": [dict(item, _score=score) for score, item in hits[:3]],
        "conflict_items": conflict_items,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "退款三天了还没到账"
    result = judge(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
