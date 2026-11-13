"""电商售后 AI 客服工作台 · 交互原型

三种数据来源：
1. mock-data.json  —— 会话、消息、知识、建议的初始数据
2. data/evaluation_results.csv —— 50 条评测集 × V1/V2/V3 的真实运行结果
3. st.session_state —— 本次运行中的操作状态（刷新即重置）

本文件是前端交互原型，不含后端、数据库与登录。详见「实现说明」页签。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

import llm
import policy
from engine import judge as engine_judge, load_kb as engine_load_kb

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mock-data.json"
EVAL_PATH = BASE_DIR / "data" / "evaluation_results.csv"

# ---------------------------------------------------------------- 基础配置

st.set_page_config(page_title="电商售后 AI 客服工作台", layout="wide", page_icon="🎧")

STYLE = """
<style>
:root{
  --ink:#1a1d23; --sub:#6b7280; --line:#e8eaee; --bg:#f7f8fa;
  --blue:#2563eb; --blue-soft:#eef3ff;
  --red:#dc2626; --red-soft:#fef2f2;
  --amber:#d97706; --amber-soft:#fffbeb;
  --green:#059669; --green-soft:#ecfdf5;
}
.block-container{padding-top:2rem;padding-bottom:1.2rem;max-width:1600px}
h1,h2,h3{color:var(--ink)}
/* 顶部指标 */
.kpi-row{display:flex;gap:10px;margin:6px 0 14px}
.kpi{flex:1;background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.kpi .n{font-size:22px;font-weight:700;color:var(--ink);line-height:1.2}
.kpi .l{font-size:12px;color:var(--sub);margin-top:2px}
.kpi.alert .n{color:var(--red)}
/* 会话头 */
.conv-head{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.conv-title{font-size:16px;font-weight:700;color:var(--ink)}
.conv-meta{font-size:12px;color:var(--sub);margin-top:4px}
.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px;border:1px solid transparent}
.chip.gray{background:#f2f3f5;color:#4b5563}
.chip.blue{background:var(--blue-soft);color:var(--blue);border-color:#dbe4ff}
.chip.red{background:var(--red-soft);color:var(--red);border-color:#fecaca}
.chip.amber{background:var(--amber-soft);color:var(--amber);border-color:#fde68a}
.chip.green{background:var(--green-soft);color:var(--green);border-color:#bbf7d0}
.tr{display:flex;gap:8px;font-size:12.5px;line-height:1.6;padding:5px 0;border-bottom:1px dashed #eef0f3}
.tr:last-child{border-bottom:0}
.tr .k{flex:0 0 72px;color:#6b7280}
.tr .v{flex:1;color:#111827}
.tr .ok{color:#059669;font-weight:600}
.tr .no{color:#dc2626;font-weight:600}
.tr .warn{color:#d97706;font-weight:600}
/* 气泡 */
.chat-wrap{background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}
.row{display:flex;margin:8px 2px}
.row.customer{justify-content:flex-start}
.row.agent{justify-content:flex-end}
.row.system{justify-content:center}
.bubble{max-width:76%;padding:9px 13px;border-radius:12px;font-size:14px;line-height:1.55;white-space:pre-wrap}
.bubble.customer{background:#f4f5f7;color:var(--ink);border-top-left-radius:4px}
.bubble.agent{background:var(--blue);color:#fff;border-top-right-radius:4px}
.bubble.system{background:transparent;color:var(--sub);font-size:12px;border:1px dashed var(--line);max-width:92%}
.meta{font-size:11px;color:#9ca3af;margin:0 6px 2px}
/* 卡片 */
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.card-h{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.card-t{font-size:13px;font-weight:700;color:var(--ink)}
.card-body{font-size:14px;line-height:1.65;color:#23262f;white-space:pre-wrap}
.conf{font-size:11px;color:var(--sub)}
.ev{border-left:3px solid #dbe4ff;padding:6px 10px;margin:8px 0;background:#fafbff;border-radius:0 6px 6px 0}
.ev-t{font-size:13px;font-weight:600;color:var(--ink)}
.ev-m{font-size:11px;color:var(--sub);margin:2px 0}
.ev-x{font-size:12px;color:#4b5563;line-height:1.5}
/* 状态条 */
.state{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--sub);margin:8px 0}
.dot{width:8px;height:8px;border-radius:50%}
/* 进度条 */
.bar{height:8px;background:#eef0f4;border-radius:999px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;border-radius:999px;background:var(--blue)}
.small{font-size:12px;color:var(--sub)}
.fact{background:#fafbfc;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.fk{font-size:11px;color:var(--sub)}
.fv{font-size:14px;font-weight:600;margin-top:2px;word-break:break-all}
.cost{margin-top:8px;font-size:12.5px;color:#374151;line-height:1.95;border-top:1px dashed var(--line);padding-top:8px;min-height:86px}
.cost b{font-weight:600;color:#111827}
.rr-d{min-height:58px}
table.mini{width:100%;border-collapse:collapse;font-size:12.5px}
table.mini th,table.mini td{padding:4px 2px;border-bottom:1px solid var(--line);text-align:left}
table.mini tr.bad td{color:var(--red)}
/* 会话列表按钮：左对齐、像列表行 */
[class*="st-key-sel_"] button{justify-content:flex-start!important;text-align:left!important;font-weight:500;padding:6px 10px}
[class*="st-key-sel_"] button p{font-size:13px}
[class*="st-key-sel_"] button[kind="secondary"]{background:#fff!important;border:1px solid #e8eaee!important}
[class*="st-key-sel_"] button[kind="secondary"] p{color:#374151!important}
[class*="st-key-sel_"] button[kind="primary"]{background:#eef3ff!important;border:1px solid #c7d7fe!important;box-shadow:none!important}
[class*="st-key-sel_"] button[kind="primary"] p{color:#1d4ed8!important;font-weight:700!important}
[class*="st-key-sel_"]{margin-bottom:-6px}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ---------------------------------------------------------------- 数据装载

STATUS_LABEL = {
    "processing": "处理中",
    "waiting_customer": "等待客户",
    "handed_off": "已转人工",
    "resolved": "已解决",
}
PRIORITY_LABEL = {"urgent": "紧急", "high": "高", "normal": "普通", "low": "低"}
SENDER_LABEL = {"customer": "顾客", "agent": "客服", "system": "系统"}
STATE_LABEL = {
    **policy.STATE_LABEL,
    # 界面独有的一个态：还没产出建议（不是判定结果）
    "missing": "暂无建议",
}
STATE_CHIP = {
    "ready": "chip green",
    "need_info": "chip amber",
    "conflict": "chip amber",
    "no_knowledge": "chip gray",
    "blocked": "chip red",
    "missing": "chip gray",
}
STATUS_CHIP = {
    "processing": "chip amber",
    "waiting_customer": "chip blue",
    "handed_off": "chip gray",
    "resolved": "chip green",
}
LIFECYCLE_DOT = {
    "pending": "#d97706",
    "adopted": "#2563eb",
    "edited": "#7c3aed",
    "sent": "#059669",
    "ignored": "#9ca3af",
    "handed_off": "#dc2626",
}
LIFECYCLE_LABEL = {
    "pending": "未处理",
    "adopted": "已采纳未发送",
    "edited": "已编辑",
    "sent": "已发送",
    "ignored": "已忽略",
    "handed_off": "已转人工",
}


@st.cache_data(show_spinner=False)
def load_mock_data() -> dict:
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner=False)
def get_engine_kb() -> list:
    return engine_load_kb()


@st.cache_data(show_spinner=False)
def load_eval_rows() -> list[dict]:
    if not EVAL_PATH.exists():
        return []
    with EVAL_PATH.open(encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def now_text() -> str:
    return datetime.now().strftime("%H:%M:%S")


def trial_block(key: str, kb: list, eval_rows: list[dict], show_expected: bool = False,
                modes: list[str] | None = None) -> None:
    """引擎试算：输入任意问题，用 DeepSeek 或规则引擎实时判定。"""
    modes = modes or ["DeepSeek 生成", "规则引擎"]
    col_q, col_v = st.columns([3, 1.1])
    with col_q:
        query = st.text_input(
            "客户问题", key=f"q_{key}", placeholder="输入一句客户问题，例如：退款三天了还没到账",
            label_visibility="collapsed",
        )
    with col_v:
        want_llm = bool(modes) and modes[0] == "DeepSeek 生成" and llm.enabled()
        st.selectbox(
            "判定方式",
            modes,
            index=0 if want_llm or not llm.enabled() else 0,
            key=f"v_{key}",
            label_visibility="collapsed",
        )
        mode = st.session_state[f"v_{key}"]

    if not query.strip():
        st.markdown(
            '<div class="small">输入任意问题，看它怎么判：意图、风险、是否转人工、命中哪条知识。'
            '可以试试「这个怎么样啊」「帮我写个小红书文案」——规则版这两条会判错，切 DeepSeek 再试一次。</div>',
            unsafe_allow_html=True,
        )
        return

    # ---------- DeepSeek 生成 ----------
    if mode == "DeepSeek 生成":
        if not llm.enabled():
            st.warning("未配置 DeepSeek key（.streamlit/secrets.toml 或环境变量 DEEPSEEK_API_KEY），请改用规则模式。")
            return
        with st.spinner("DeepSeek 正在读知识库并生成…"):
            llm_res = llm.generate_suggestion(
                [{"from": "customer", "text": query}], kb, None
            )
        if llm_res.get("state") == "error":
            st.error(f"调用失败，已保留规则结果可用：{llm_res.get('reason','')[:120]}")
            return

        lstate = llm_res["state"]
        kb_index = {item["id"]: item for item in kb}
        st.markdown(
            f"""
<div class="card">
  <div class="card-h"><div class="card-t">DeepSeek 判定</div>
    <span class="{STATE_CHIP.get(lstate, 'chip gray')}">{STATE_LABEL.get(lstate, lstate)}</span></div>
  <div class="small"><span class="chip blue">DeepSeek 生成</span>　
    模型输出：{lstate}　引用 {len(llm_res.get('used_kb_ids') or [])} 条知识</div>
  <div class="card-body" style="margin-top:6px">{llm_res['answer'] or '（不生成结论）'}</div>
  <div class="conf">{llm_res.get('reason','')}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        used = [kb_index[i] for i in (llm_res.get("used_kb_ids") or []) if i in kb_index]
        if used:
            with st.expander(f"它引用的知识（{len(used)} 条）", expanded=False):
                for item in used:
                    st.markdown(
                        f'<div class="ev"><div class="ev-t">{item["id"]}　{item["question"]}</div>'
                        f'<div class="ev-m">适用 {item.get("conditions","")} ｜ 风险 {item["risk_level"]}</div>'
                        f'<div class="ev-x">{item["answer"]}</div></div>',
                        unsafe_allow_html=True,
                    )
        st.markdown(
            '<div class="small">DeepSeek 结果不参与那 50 条标注对照——它没跑过全量评测，'
            '所以只用来展示生成效果，不产出准确率。</div>',
            unsafe_allow_html=True,
        )
        return

    # ---------- 规则引擎 ----------
    version = "V3" if mode.startswith("规则引擎") else mode.replace("规则 ", "")
    res = engine_judge(query, kb, version)
    state = res["state"]
    st.markdown(
        f"""
<div class="card">
  <div class="card-h"><div class="card-t">判定结果</div>
    <span class="{STATE_CHIP.get(state, 'chip gray')}">{STATE_LABEL.get(state, state)}</span></div>
  <div class="small">
    意图 <b>{res['intent']}</b>　风险 <b>{res['risk_level']}</b>　
    转人工 <b>{'是' if res['handoff'] else '否'}</b>　版本 {version}
  </div>
  <div class="card-body" style="margin-top:6px">{res['answer'] or '（不生成结论）'}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    hits = res["hits"]
    if hits:
        with st.expander(f"命中知识（{len(hits)} 条，按匹配度排序）", expanded=False):
            for item in hits:
                st.markdown(
                    f'<div class="ev"><div class="ev-t">{item["id"]}　{item["question"]}</div>'
                    f'<div class="ev-m">匹配度 {item["_score"]} ｜ 风险 {item["risk_level"]} '
                    f'｜ {item.get("conditions", "")}</div>'
                    f'<div class="ev-x">{item["answer"]}</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="small">无命中知识 → 判定为知识不足，不生成结论。</div>', unsafe_allow_html=True)

    if res["conflict_items"]:
        ids = "、".join(i["id"] for i in res["conflict_items"])
        if state == "conflict":
            st.markdown(
                f'<div class="small" style="color:#d97706">命中互斥组（{ids}），两条规则结论相反 → '
                f'判定为规则冲突：并列展示，不替客服选择。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="small" style="color:#9ca3af">同时命中互斥组（{ids}），但按优先级'
                f'「高风险 &gt; 冲突」，本次先拦为高风险。</div>',
                unsafe_allow_html=True,
            )

    if show_expected and eval_rows:
        matched = [r for r in eval_rows if r["user_query"].strip() == query.strip()]
        if matched:
            exp = next((r for r in matched if r["version"] == version), matched[0])
            ok_i = exp["pred_intent"] == exp["expected_intent"]
            ok_h = truthy(exp["pred_handoff"]) == truthy(exp["expected_handoff"])
            st.markdown(
                f'<div class="ev"><div class="ev-t">这条在评测集里（{exp["version"]}）</div>'
                f'<div class="ev-x">标注：意图 {exp["expected_intent"]}、转人工 '
                f'{"是" if truthy(exp["expected_handoff"]) else "否"}<br>'
                f'引擎：意图 {exp["pred_intent"]} {"✓" if ok_i else "✗"}、转人工 '
                f'{"是" if truthy(exp["pred_handoff"]) else "否"} {"✓" if ok_h else "✗"}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="small">这条不在 50 条评测集里，没有标注可对照。</div>', unsafe_allow_html=True)


data = load_mock_data()
conversations: list[dict] = data.get("conversations", [])
messages: list[dict] = data.get("messages", [])
suggestions: list[dict] = data.get("suggestions", [])
handoff_reasons: list[dict] = data.get("handoffReasons", [])

SUGGESTION_BY_CONV = {item.get("convId"): item for item in suggestions}
MESSAGES_BY_CONV: dict[str, list[dict]] = {}
for _m in messages:
    MESSAGES_BY_CONV.setdefault(_m.get("convId", ""), []).append(_m)

# ---------------------------------------------------------------- 会话运行态

if "conv_state" not in st.session_state:
    st.session_state.conv_state = {
        c["id"]: {
            "status": c.get("status", "processing"),
            "lifecycle": "pending",
            "adopted": False,
            "edited": False,
            "final_text": "",
            "sent": [],
            "handoff_reason": "",
            "outcome": "",
            "audit": [],
        }
        for c in conversations
    }


def log_action(cid: str, action: str, detail: str = "") -> None:
    st.session_state.conv_state[cid]["audit"].append(
        {"at": now_text(), "action": action, "detail": detail}
    )


def state_of(cid: str) -> dict:
    return st.session_state.conv_state[cid]


def label(c: dict) -> str:
    st_ = state_of(c["id"])
    risk = _has_risk(c)
    unread = bool(c.get("unread")) and c["id"] not in st.session_state.get("read_set", set())
    mark = "🔴" if risk else ("🔵" if unread else "⚪")
    return (
        f"{mark} {c.get('customer',{}).get('name','未知')} · {c.get('topic','未分类')}"
        f" · {STATUS_LABEL.get(st_['status'], st_['status'])}"
    )


# 风险词只有一份：engine.HANDOFF_KEYWORDS（16 个，11 high / 5 medium）。
# 这里曾经另写过一个 8 词列表，且早已无人使用——留着只会让文档引错口径，故改为别名。
RISK_WORDS = policy.HANDOFF_KEYWORDS


@st.cache_data(show_spinner=False)
def load_orders() -> dict:
    return policy.load_orders()


def customer_texts(cid: str) -> list[str]:
    return [m.get("text", "") for m in MESSAGES_BY_CONV.get(cid, []) if m.get("from") == "customer"]


def conv_as_of(cid: str) -> str:
    """模拟数据没有真实时钟：以该会话最后一条消息的时间作为判定参考时刻。

    话术里的"停了几天"必须由它减订单字段算出来。不用 datetime.now()——
    那会让同一张单在不同日期跑出不同结论，演示就没法复现了。
    """
    return max((m.get("at", "") for m in MESSAGES_BY_CONV.get(cid, [])), default="")


def order_as_of(order_no: str) -> str:
    """「试一句」选了订单就用该单所属会话的时刻，没选就用全局最后一条消息。"""
    if order_no:
        for o in load_orders().get("orders") or []:
            if o.get("orderId") == order_no:
                t = conv_as_of(o.get("convId", ""))
                if t:
                    return t
    return max((m.get("at", "") for m in messages), default="")


def get_verdict(conv: dict) -> policy.Verdict:
    """判定只发生一次，结果进缓存。

    建议卡、发送门禁、转人工原因、审计全部读这一份——
    以前是两处各写一套 if，才会出现"卡片说不可发、按钮却能点"。
    """
    cid = conv["id"]
    cache = st.session_state.setdefault("verdict_cache", {})
    if cid not in cache:
        cust = customer_texts(cid)
        # 用整段客户发言判定，不只用最后一句：客户常常先提问、再补一句催促，
        # 只看最后一句会把"退款三天没到账"读成"麻烦快点"。
        question = " ".join(cust) if cust else (conv.get("topic") or "")
        cache[cid] = policy.decide(
            question,
            order_no=conv.get("orderId", ""),
            kb=get_engine_kb(),
            orders_doc=load_orders(),
            dialog=question,
            as_of=conv_as_of(cid),
        )
    return cache[cid]


def get_suggestion(conv: dict) -> dict:
    """把 Verdict 翻译成界面要的字段。

    注意：LLM 只负责措辞。它的 state 一律丢弃——判定权只在 policy.decide。
    """
    cid = conv["id"]
    cache = st.session_state.setdefault("sug_cache", {})
    if cid in cache:
        return cache[cid]

    v = get_verdict(conv)
    evidences = [
        {
            "kbId": e.get("kbId", ""),
            "title": e.get("title", ""),
            "excerpt": e.get("excerpt", ""),
            "scope": e.get("scope", ""),
            "risk": e.get("risk", ""),
            "updatedAt": e.get("updatedAt", ""),
        }
        for e in v.evidence
    ]

    text, source, note = "", "policy", ""
    if v.state == "ready":
        # 用订单字段拼一句话，而不是把知识库原文当答复
        text, src = policy.compose_answer(v)
        source = "order" if src == "order" else "kb"
        note = "已用订单字段重写" if src == "order" else "引用规则原文（本条没有可用字段）"
        if llm.enabled():
            cand = [item for item in get_engine_kb() if item["id"] in {e["kbId"] for e in evidences}]
            cust = [m for m in MESSAGES_BY_CONV.get(cid, []) if m.get("from") == "customer"]
            with st.spinner("DeepSeek 正在把判定写成话术…"):
                res = llm.generate_suggestion(
                    cust, cand, {"orderId": conv.get("orderId", ""), "channel": conv.get("channel", "")}
                )
            if res.get("state") == "ready" and res.get("answer"):
                text, source, note = res["answer"], "llm", res.get("reason", "")

    sug = {
        "state": v.state,
        "text": text,
        "conflictNote": "；".join(v.notes),
        "blockedNote": v.why,
        "why": v.why,
        "owner": v.owner,
        "missing": [{"field": m.field, "ask": m.ask} for m in v.missing],
        "actions": v.actions,
        "evidences": evidences,
        "risks": v.risks,
        "can_send": v.can_send,
        "requires_ack": v.requires_ack,
        "handoff_reason": v.handoff_reason,
        "order_no": v.order_no,
        "order": v.order,
        "intent": v.intent,
        "question": v.question,
        "steps": v.steps,
        "source": source,
        "reason": note,
    }
    cache[cid] = sug
    return sug


def _field_line(order: dict) -> str:
    """把这单里真被用到的几个字段摊开，让人看见"结论是从数据算的"。"""
    if not order:
        return ""
    bits = []
    if order.get("title"):
        bits.append(f'商品「{order["title"]}」')
    logistics = order.get("logistics") or {}
    if logistics.get("status"):
        bits.append(f'物流「{logistics["status"]}」')
    after = order.get("afterSale") or {}
    if after:
        bits.append("未拆封" if after.get("opened") is False else "已拆封")
        bits.append("未使用" if after.get("used") is False else "已使用")
    refund = order.get("refund") or {}
    if refund:
        bits.append("退款单号已生成" if refund.get("refundNo") else "退款单号为空")
    return "　".join(bits)


def trace_html(v: dict) -> str:
    """渲染 policy.decide() 真实走过的步骤。

    这些行由判定层在每一个分支里 push 进来，界面只负责显示——
    早先这一块是"照着结论补写"的，所以会出现"没查单却写查无此单"，
    看着像审计日志，其实是另一个模板。
    """
    steps = v.get("steps") or []
    rows = []
    for name, result in steps:
        bad = any(k in str(result) for k in ("为空", "查无此单", "没有订单", "0 条", "未命中", "high"))
        cls = "no" if bad else ""
        rows.append((name, f'<span class="{cls}">{result}</span>' if cls else result))
    rows.append(("谁接手", v.get("owner") or "—"))
    return '<div class="card" style="padding:10px 12px">' + "".join(
        f'<div class="tr"><div class="k">{k}</div><div class="v">{val}</div></div>'
        for k, val in rows
    ) + "</div>"


def render_try_box() -> None:
    """默认屏上的自由输入：换个问题/换张单，结论必须跟着变。"""
    DEMO_Q = "我的快递五天没动了，一直显示在转运中心"
    DEMO_ORDER = "SO20261102099"
    with st.expander("试一句：输入客户问题，现场判定（不是预设剧本）", expanded=True):
        c1, c2 = st.columns([3, 1.15])
        question = c1.text_input(
            "客户问题",
            key="try_q",
            placeholder="例如：退款三天了还没到账 / 你们能开发票吗 / 我要退一赔三",
            label_visibility="collapsed",
        )
        order_ids = ["（不选订单）"] + [
            o.get("orderId", "") for o in (load_orders().get("orders") or [])
        ]
        default_idx = order_ids.index(DEMO_ORDER) if DEMO_ORDER in order_ids else 0
        picked = c2.selectbox(
            "订单号", order_ids, index=default_idx, key="try_order", label_visibility="collapsed"
        )
        if not question.strip():
            st.markdown(
                '<div class="small">不填也能看：下面是这句默认问题 + 这张单跑出来的一次真判定。</div>',
                unsafe_allow_html=True,
            )
            question = DEMO_Q
        v = policy.decide(
            question.strip(),
            order_no="" if picked.startswith("（") else picked,
            kb=get_engine_kb(),
            orders_doc=load_orders(),
            dialog=question.strip(),
            as_of=order_as_of("" if picked.startswith("（") else picked),
        )
        st.markdown(
            f'<div class="card"><div class="card-h"><div class="card-t">判定结果</div>'
            f'<span class="{STATE_CHIP.get(v.state, "chip gray")}">{STATE_LABEL.get(v.state, v.state)}</span></div>'
            f'<div class="card-body">{(v.why or "").replace("<", "&lt;")}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            trace_html({"steps": v.steps, "owner": v.owner}),
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- 回调

def cb_select(cid: str) -> None:
    st.session_state["selected_conv"] = cid
    st.session_state.setdefault("read_set", set()).add(cid)


def cb_adopt(cid: str, text: str) -> None:
    st.session_state[f"draft_{cid}"] = text
    state_of(cid)["adopted"] = True
    state_of(cid)["lifecycle"] = "adopted"
    log_action(cid, "采纳建议", "将 AI 建议原文填入回复框")


def cb_send(cid: str, suggestion_text: str) -> None:
    draft = (st.session_state.get(f"draft_{cid}") or "").strip()
    if not draft:
        return
    st_ = state_of(cid)
    changed = draft != (suggestion_text or "").strip()
    st_["final_text"] = draft
    st_["edited"] = changed
    st_["lifecycle"] = "edited" if changed else "sent"
    st_["sent"].append({"from": "agent", "text": draft, "at": now_text()})
    st_["status"] = "waiting_customer"
    # 发完就清空草稿，并让门禁挡住第二次点击——否则同一句话会连发两条。
    st.session_state[f"draft_{cid}"] = ""
    has_sugg = bool((suggestion_text or "").strip())
    if not changed and has_sugg:
        # 预填的建议就算没点「采纳」，原样发出去也属于采纳——
        # 否则采纳率把"直接发送"的人漏掉了。
        st_["adopted"] = True
        st_["outcome"] = "AI 建议原样采纳"
    elif changed and has_sugg:
        st_["outcome"] = "人工改写后发送"
    else:
        st_["outcome"] = "人工撰写发送"
    log_action(cid, "发送", "改写后发送" if changed else "原样发送")


def cb_ignore(cid: str) -> None:
    st_ = state_of(cid)
    st_["lifecycle"] = "ignored"
    st_["outcome"] = "驳回 AI 建议"
    log_action(cid, "忽略建议", "客服未采用 AI 建议")


def cb_handoff(cid: str, reason: str) -> None:
    st_ = state_of(cid)
    st_["status"] = "handed_off"
    st_["lifecycle"] = "handed_off"
    st_["handoff_reason"] = reason
    st_["outcome"] = f"人工接管（{reason}）"
    log_action(cid, "转人工", reason)


def cb_resolve(cid: str) -> None:
    st_ = state_of(cid)
    st_["status"] = "resolved"
    if st_["lifecycle"] in {"pending", "adopted"}:
        st_["lifecycle"] = "sent"
    log_action(cid, "标记已解决")


def cb_reopen(cid: str) -> None:
    st_ = state_of(cid)
    st_["status"] = "processing"
    log_action(cid, "重新打开会话")


# ---------------------------------------------------------------- A 场景：退货原因

REASON_SCRIPT = {
    "seven_day": "您好，这笔订单未拆封，签收后 7 天内可以申请无理由退货。寄回运费由您承担，我们收到验收后原路退款。",
    "exchange": "您好，可以给您换同款 02 自然色。签收后 15 天内可申请；非质量问题寄回运费由您承担，仓库收到寄回后 48 小时内安排发出。",
}


def render_return_reason() -> None:
    """A 场景：同一单，三种退货原因，商家的代价不一样。

    这一屏不讲"AI 会不会说话"，只讲"选错原因谁付钱"。三个原因都提交得上去，
    但只有品质退货会被平台记成商责——而它恰好是客服最省事的那个。
    """
    doc = load_orders()
    order = next((o for o in doc.get("orders", []) if str(o.get("purpose", "")).startswith("A")), None)
    if not order:
        st.warning("找不到 A 场景订单（orders.json 里 purpose 以 A 开头的那张）。")
        return

    opts = {o.code: o for o in policy.return_reasons(order)}
    after = order.get("afterSale") or {}
    spec = order.get("spec") or {}

    st.markdown("### 同一单，三种退货原因，代价不一样")
    st.markdown(
        '<div class="small">客户只是买错了色号。三个原因在后台<b>都提交得上去</b>，'
        '但只有品质退货会让商家多掏来回运费、还被计入商责退货率。'
        '本系统<b>禁掉品质退货</b>（这单没有质量证据）；七天无理由与换货<b>都可选</b>，'
        '它们的差别是运费谁承担——所以这一屏不是"只有一个正确答案"，是"选错谁付钱"。</div>',
        unsafe_allow_html=True,
    )

    # ---------- 上面一条事实带 ----------
    facts = [
        ("订单号", order.get("orderId", "—")),
        ("商品", order.get("title", "—")),
        ("下单色号", spec.get("色号") or "—"),
        ("客户想要", "02 自然色"),
        ("签收", str(order.get("deliveredAt") or "—")[:10]),
        ("拆封 / 使用", ("已拆封" if after.get("opened") else "未拆封") + " · " + ("已使用" if after.get("used") else "未使用")),
    ]
    cols = st.columns(len(facts))
    for col, (k, v) in zip(cols, facts):
        col.markdown(
            f'<div class="fact"><div class="fk">{k}</div><div class="fv">{v}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="card" style="margin-top:12px"><div class="card-body">'
        f'客户原话：「{after.get("reason") or "—"}」</div>'
        f'<div class="conf">售后类型：{after.get("type") or "未提交"}　·　'
        f'申请时间：{str(after.get("appliedAt") or "—")}</div></div>',
        unsafe_allow_html=True,
    )

    # ---------- 三个原因并排 ----------
    if "rr_choice" not in st.session_state:
        st.session_state["rr_choice"] = "exchange"

    st.markdown('<div class="card-t" style="margin:14px 0 6px">选哪个原因，决定这单记在谁账上</div>', unsafe_allow_html=True)
    card_cols = st.columns(3, gap="medium")
    for col, code in zip(card_cols, ["seven_day", "exchange", "quality_return"]):
        opt = opts[code]
        with col:
            with st.container(border=True):
                badge = '<span class="chip green">可用</span>' if opt.allowed else '<span class="chip red">不可用</span>'
                sel = st.session_state["rr_choice"] == code
                if sel:
                    badge += ' <span class="chip blue">当前选择</span>'
                st.markdown(f"**{opt.label}**　{badge}", unsafe_allow_html=True)
                st.markdown(f'<div class="small rr-d">{opt.why}</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="cost">'
                    f'<div>运费　<b>{opt.cost_text.replace("商家多掏", "商家多掏 ")}</b></div>'
                    f'<div>品退率　<b>{"计入商责退货率" if opt.counts_return_rate else "不计入"}</b></div>'
                    f'<div>店铺评分　<b>{"计入考核" if opt.affects_score else "不影响"}</b></div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "选这个" if opt.allowed else "选不了",
                    key=f"rr_{code}",
                    disabled=not opt.allowed,
                    type="primary" if sel else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["rr_choice"] = code
                    st.rerun()

    # ---------- 选中之后的判定 ----------
    chosen = opts[st.session_state["rr_choice"]]
    st.markdown('<div class="card-t" style="margin:16px 0 6px">这一步的判定</div>', unsafe_allow_html=True)
    vcol1, vcol2 = st.columns([1.5, 1.0], gap="medium")
    with vcol1:
        if chosen.allowed and chosen.code in REASON_SCRIPT:
            st.markdown(
                f'<div class="card"><div class="card-h"><div class="card-t">可以这样回复 · {chosen.label}</div>'
                f'<span class="chip green">可发</span></div>'
                f'<div class="card-body">{REASON_SCRIPT[chosen.code]}</div>'
                f'<div class="conf">依据：{"售后规则-001 + 售后规则-002" if chosen.code == "seven_day" else "售后规则-005 可以换货吗？"}　·　'
                f'兜底：客服</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="card"><div class="card-h"><div class="card-t">不生成话术</div>'
                '<span class="chip amber">需先补证据</span></div>'
                '<div class="card-body">后台确实能提交这个原因，但本系统不替你选。'
                '要按品质退货走，必须先有两样东西：① 客户提供的质检报告或医院证明；② 客户书面确认这不是拍错色号。'
                '两样都没有时，这一格只说明后果——提交了就会被记成商责退货，运费由商家承担。</div>'
                '<div class="conf">依据：订单 policyHooks K2（美妆个护质量问题需医院证明或质检报告）＋ 售后规则-003（过敏类必须转人工）　·　兜底：客服组长（升级才可提交）</div></div>',
                unsafe_allow_html=True,
            )
    with vcol2:
        st.markdown(
            '<div class="card"><div class="card-t" style="margin-bottom:8px">三种选法的账</div>'
            '<table class="mini">'
            '<tr><th>原因</th><th>商家多掏</th><th>品退率</th></tr>'
            f'<tr><td>七天无理由</td><td>¥0</td><td>不计</td></tr>'
            f'<tr><td>换货</td><td>¥{opts["exchange"].seller_cost}</td><td>不计</td></tr>'
            f'<tr class="bad"><td>品质退货</td><td>¥{opts["quality_return"].seller_cost}</td><td>计入</td></tr>'
            "</table>"
            '<div class="conf">运费按单程 12 元假设，不是实测数字；品退率考核引 E-07/E-06。</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="small" style="margin-top:6px">'
        "这一屏要证明的只有一件事：<b>把客户的问题归错类，代价不会消失，只会转移给商家</b>——"
        "转移到品退率和下个月的处罚里。</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- 表头与统计

pending = sum(1 for c in conversations if state_of(c["id"])["status"] == "processing")
waiting = sum(1 for c in conversations if state_of(c["id"])["status"] == "waiting_customer")
def _has_risk(c: dict) -> bool:
    """风险数读实时判定，不读 JSON 里的旧标记。"""
    v = get_verdict(c)
    return bool(v.risks) or v.state == "blocked"


risk_conv = sum(1 for c in conversations if _has_risk(c))
resolved = sum(1 for c in conversations if state_of(c["id"])["status"] == "resolved")
handed = sum(1 for c in conversations if state_of(c["id"])["status"] == "handed_off")

st.markdown("## 电商售后 AI 客服工作台")
st.markdown(
    '<div class="small">只给建议、不替人做决定：信息不足、知识不足、高风险、规则冲突这四种情况下，它不给话术，只给动作（追问 / 转人工 / 禁止发送）。</div>',
    unsafe_allow_html=True,
)

show_meta = (
    str(st.query_params.get("dev", "0")) == "1"
    or str(st.query_params.get("llm", "0")) == "1"
)

if show_meta:
    tab_work, tab_reason, tab_live, tab_eval, tab_note = st.tabs(
        ["🖥 工作台", "🧾 退货原因（A 场景）", "🤖 实时问答（引擎）", "📊 评测概览", "📄 实现说明"]
    )
else:
    # 演示模式：只留两个真页面。实时问答、评测、实现说明都是「讲给人听」的材料，
    # 放在默认界面会让人不知道该看哪，统一收到 ?dev=1 下面。
    tab_work, tab_reason = st.tabs(["🖥 工作台", "🧾 退货原因"])
    tab_live = tab_eval = tab_note = None

# ================================================================ A 场景：退货原因
with tab_reason:
    render_return_reason()

# ================================================================ 工作台
with tab_work:
    st.markdown(
        f"""
<div class="kpi-row">
  <div class="kpi"><div class="n">{len(conversations)}</div><div class="l">会话总数</div></div>
  <div class="kpi{' alert' if pending else ''}"><div class="n">{pending}</div><div class="l">待处理</div></div>
  <div class="kpi"><div class="n">{waiting}</div><div class="l">等待客户</div></div>
  <div class="kpi"><div class="n">{handed}</div><div class="l">已转人工</div></div>
  <div class="kpi"><div class="n">{resolved}</div><div class="l">已解决</div></div>
</div>
<div class="small" style="margin:-8px 0 10px">
  上面四格互斥，合计 {pending + waiting + handed + resolved} 条 = 会话总数；
  其中含风险 <b>{risk_conv}</b> 条（风险与状态是两回事，不叠加进那四格）。
</div>
""",
        unsafe_allow_html=True,
    )
    render_try_box()

    left, mid, right = st.columns([1.05, 2.0, 1.2], gap="medium")

    # ---------- 左：固定会话列表 ----------
    with left:
        st.markdown('<div class="card-t">会话列表</div>', unsafe_allow_html=True)
        keyword = st.text_input("搜索", placeholder="客户名 / 订单号 / 会话编号", label_visibility="collapsed")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            status_filter = st.selectbox(
                "状态", ["全部", "处理中", "等待客户", "已转人工", "已解决"], index=0, label_visibility="collapsed"
            )
        with fcol2:
            risk_only = st.checkbox("仅看风险", value=False)

        def match(c: dict) -> bool:
            st_ = state_of(c["id"])
            if status_filter != "全部" and STATUS_LABEL.get(st_["status"], st_["status"]) != status_filter:
                return False
            if risk_only and not _has_risk(c):
                return False
            if keyword.strip():
                hay = f"{c.get('id','')}{c.get('orderId','')}{c.get('customer',{}).get('name','')}{c.get('topic','')}"
                if keyword.strip() not in hay:
                    return False
            return True

        shown = [c for c in conversations if match(c)]
        st.markdown(f'<div class="small">共 {len(shown)} / {len(conversations)} 条</div>', unsafe_allow_html=True)

        if "selected_conv" not in st.session_state:
            st.session_state["selected_conv"] = conversations[0]["id"] if conversations else None

        if not shown:
            st.info("没有符合条件的会话")
        else:
            scope_ids = {c["id"] for c in shown}
            if st.session_state["selected_conv"] not in scope_ids:
                st.session_state["selected_conv"] = shown[0]["id"]

            box = st.container(height=430)
            with box:
                for c in shown:
                    cid = c["id"]
                    is_sel = cid == st.session_state["selected_conv"]
                    st.button(
                        label(c),
                        key=f"sel_{cid}",
                        type="primary" if is_sel else "secondary",
                        use_container_width=True,
                        on_click=cb_select,
                        args=(cid,),
                    )

        selected_id = st.session_state.get("selected_conv")

        st.markdown(
            '<div class="small">🔴 有风险　⚪ 无风险　🔵 有未读</div>', unsafe_allow_html=True
        )

    # ---------- 中：对话 ----------
    conv = next((c for c in conversations if c["id"] == selected_id), None)
    with mid:
        if conv is None:
            st.info("请选择一条会话")
        else:
            st_ = state_of(conv["id"])
            sug = get_suggestion(conv)
            sug_state = (sug or {}).get("state", "missing")
            customer = conv.get("customer", {})
            st.markdown(
                f"""
<div class="conv-head">
  <div class="conv-title">{customer.get('name','未知客户')} · {conv.get('topic','未分类')}</div>
  <div class="conv-meta">
    <span class="chip gray">会话 {conv.get('id')}</span>
    <span class="chip gray">订单 {conv.get('orderId','—')}</span>
    <span class="{STATUS_CHIP.get(st_['status'], 'chip gray')}">{STATUS_LABEL.get(st_['status'], st_['status'])}</span>
    <span class="chip {'red' if conv.get('priority') in {'urgent','high'} else 'gray'}">优先级 {PRIORITY_LABEL.get(conv.get('priority'), conv.get('priority'))}</span>
    <span class="{STATE_CHIP.get(sug_state)}">AI {STATE_LABEL.get(sug_state)}</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            chat = st.container(height=360)
            with chat:
                items = list(MESSAGES_BY_CONV.get(conv["id"], [])) + list(st_["sent"])
                if not items:
                    st.markdown('<div class="small">暂无消息</div>', unsafe_allow_html=True)
                for msg in items:
                    who = msg.get("from", "system")
                    text = (msg.get("text") or "").replace("<", "&lt;")
                    if who == "system":
                        html = f'<div class="row system"><div class="bubble system">{text}</div></div>'
                    else:
                        side = "customer" if who == "customer" else "agent"
                        html = (
                            f'<div class="row {side}"><div class="bubble {side}">{text}</div></div>'
                        )
                    st.markdown(html, unsafe_allow_html=True)

            # 回复区
            draft_key = f"draft_{conv['id']}"
            if draft_key not in st.session_state:
                st.session_state[draft_key] = (sug or {}).get("text", "") if sug_state == "ready" else ""

            locked = st_["status"] in {"resolved"}
            verdict_can_send = bool((sug or {}).get("can_send"))
            needs_ack = bool((sug or {}).get("requires_ack"))
            draft = st.text_area(
                "回复内容",
                key=draft_key,
                height=110,
                placeholder="AI 建议可用时会自动填入；也可以由客服改写。",
                disabled=locked,
            )
            ack = True
            if needs_ack:
                ack = st.checkbox(
                    "我已核对两条规则的原文，确认按其中一条回复",
                    key=f"ack_{conv['id']}",
                )
            # 门禁只读 verdict，不再自己判断——这是"统一成一条线"的落点
            can_send = (
                bool(draft.strip())
                and not locked
                and verdict_can_send
                and st_["status"] not in {"handed_off", "waiting_customer"}
                and ack
            )

            b1, b2, b3 = st.columns([1, 1, 1])
            with b1:
                st.button(
                    "采纳并填入",
                    key=f"adopt_{conv['id']}",
                    on_click=cb_adopt,
                    args=(conv["id"], (sug or {}).get("text", "")),
                    disabled=sug_state != "ready" or locked,
                    use_container_width=True,
                )
            with b2:
                st.button(
                    "发送回复",
                    key=f"send_{conv['id']}",
                    type="primary",
                    on_click=cb_send,
                    args=(conv["id"], (sug or {}).get("text", "")),
                    disabled=not can_send,
                    use_container_width=True,
                )
            with b3:
                st.button(
                    "忽略建议",
                    key=f"ignore_{conv['id']}",
                    on_click=cb_ignore,
                    args=(conv["id"],),
                    disabled=sug_state == "missing" or locked,
                    use_container_width=True,
                )

            if not verdict_can_send:
                tone = "#dc2626" if sug_state == "blocked" else "#b45309"
                st.markdown(
                    f'<div class="small" style="color:{tone}">'
                    f'{STATE_LABEL.get(sug_state, sug_state)}：发送已禁用。'
                    f'{sug.get("why", "")}　→　该由 {sug.get("owner", "—")} 接手。</div>',
                    unsafe_allow_html=True,
                )
            elif needs_ack and not ack:
                st.markdown(
                    '<div class="small" style="color:#b45309">规则冲突：勾选确认后才能发送。</div>',
                    unsafe_allow_html=True,
                )
            elif st_["status"] == "waiting_customer":
                st.markdown(
                    '<div class="small" style="color:#2563eb">本条已发送，等待客户回复；'
                    '客户回复后才会重新生成建议。</div>',
                    unsafe_allow_html=True,
                )
            elif st_["status"] == "handed_off":
                st.markdown(
                    '<div class="small" style="color:#6b7280">已转人工，本会话不再由 AI 建议驱动。</div>',
                    unsafe_allow_html=True,
                )

            reasons = [r.get("label", "其他") for r in handoff_reasons] or ["其他"]
            reason_codes = [r.get("code", "") for r in handoff_reasons] or ["other"]
            default_code = (sug or {}).get("handoff_reason") or {
                "no_knowledge": "no_knowledge",
                "conflict": "conflict",
                "blocked": "high_risk",
            }.get(sug_state, "customer_request")
            default_index = reason_codes.index(default_code) if default_code in reason_codes else 0
            reason = st.selectbox(
                "转人工原因", reasons, index=default_index, key=f"reason_{conv['id']}", disabled=locked
            )
            h1, h2 = st.columns([1, 1])
            with h1:
                st.button(
                    "转人工处理",
                    key=f"handoff_{conv['id']}",
                    on_click=cb_handoff,
                    args=(conv["id"], reason),
                    disabled=locked or st_["status"] == "handed_off",
                    use_container_width=True,
                )
            with h2:
                if st_["status"] == "resolved":
                    st.button(
                        "重新打开", key=f"reopen_{conv['id']}", on_click=cb_reopen,
                        args=(conv["id"],), use_container_width=True,
                    )
                else:
                    st.button(
                        "标记已解决", key=f"resolve_{conv['id']}", on_click=cb_resolve,
                        args=(conv["id"],), disabled=locked, use_container_width=True,
                    )

    # ---------- 右：AI 建议卡 ----------
    with right:
        if conv is None:
            st.info("—")
        else:
            st_ = state_of(conv["id"])
            sug = get_suggestion(conv)
            sug_state = (sug or {}).get("state", "missing")
            conf = (sug or {}).get("confidence", 0) or 0

            if sug is None:
                st.markdown(
                    '<div class="card"><div class="card-h"><div class="card-t">AI 建议</div>'
                    '<span class="chip gray">暂无</span></div><div class="card-body">该会话没有生成建议。</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                body = ""
                if sug_state == "ready":
                    body = (sug.get("text") or "").replace("<", "&lt;")
                elif sug_state == "need_info":
                    missed = sug.get("missing") or []
                    body = "<br>".join(
                        f'缺「{m["field"]}」：{m["ask"]}'.replace("<", "&lt;") for m in missed
                    ) or (sug.get("why") or "信息不足，先补齐再答。").replace("<", "&lt;")
                elif sug_state == "conflict":
                    body = (sug.get("conflictNote") or "检索到结论相反的规则，需人工判断。").replace("<", "&lt;")
                elif sug_state == "no_knowledge":
                    # 只承诺界面上真有的动作：转人工按钮存在，知识缺口登记还没做。
                    body = "未检索到适用规则，不生成结论。可转人工；知识缺口登记尚未实现。"
                elif sug_state == "blocked":
                    body = (sug.get("blockedNote") or "检测到高风险，已禁止生成可发送内容。").replace("<", "&lt;")

                source = sug.get("source", "script")
                source_chip = {
                    "llm": '<span class="chip blue">DeepSeek 改写</span>',
                    "order": '<span class="chip green">订单字段拼的</span>',
                    "kb": '<span class="chip gray">规则原文</span>',
                }.get(source, '<span class="chip gray">判定层输出</span>')
                reason = (sug.get("reason") or "").replace("<", "&lt;")
                st.markdown(
                    f"""
<div class="card">
  <div class="card-h">
    <div class="card-t">AI 建议</div>
    <span class="{STATE_CHIP.get(sug_state)}">{STATE_LABEL.get(sug_state)}</span>
  </div>
  <div class="card-body">{body}</div>
  <div class="conf">{source_chip}{f'　·　{reason}' if reason and source == 'llm' else ''}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="small" style="margin:-4px 0 8px">'
                    f'判定：<b>{STATE_LABEL.get(sug_state, sug_state)}</b>　'
                    f'归属：{sug.get("owner", "—")}　'
                    f'为什么：{sug.get("why", "")}'
                    f'{"　可发" if sug.get("can_send") else "　不可发"}</div>',
                    unsafe_allow_html=True,
                )

                counterfactual = (conv.get("counterfactual") or "").strip()
                if counterfactual:
                    # 最有说服力的不是“AI 答了什么”，而是“没有这道判定会发生什么”。
                    st.markdown(
                        '<div class="small" style="margin:-2px 0 8px;padding:6px 9px;'
                        'background:#fffbeb;border-left:3px solid #d97706;border-radius:4px;color:#92400e">'
                        f'如果没有这道判定：{counterfactual.replace("<", "&lt;")}</div>',
                        unsafe_allow_html=True,
                    )

                with st.expander("它是怎么判出来的（判定层真实步骤）", expanded=True):
                    st.markdown(
                        trace_html({"steps": sug.get("steps") or [], "owner": sug.get("owner", "")}),
                        unsafe_allow_html=True,
                    )

                if source == "llm":
                    if st.button("重新生成", key=f"regen_{conv['id']}", use_container_width=True):
                        st.session_state.get("sug_cache", {}).pop(conv["id"], None)
                        st.rerun()

                ai_risks = sug.get("risks") or []
                if ai_risks:
                    # 只展示实时判定结果。mock-data 里的 riskFlags/S1–S4 是旧版遗留，
                    # 不再参与界面判断，避免“KPI 说有风险、筛选说没有”。
                    parts = ['<div class="card"><div class="card-h"><div class="card-t">风险提示</div>'
                             f'<span class="chip red">{len(ai_risks)} 条</span></div>']
                    for r in ai_risks:
                        color = "#dc2626" if r.get("level") == "high" else "#d97706"
                        parts.append(
                            f'<div class="ev" style="border-left-color:{color}">'
                            f'<div class="ev-t">{r.get("level","medium").upper()} 风险</div>'
                            f'<div class="ev-x">{r.get("reason","")}</div></div>'
                        )
                    parts.append("</div>")
                    st.markdown("".join(parts), unsafe_allow_html=True)

                evidences = sug.get("evidences") or []
                with st.expander(f"知识依据（{len(evidences)} 条）", expanded=True):
                    if not evidences:
                        st.markdown('<div class="small">该建议没有关联知识依据。</div>', unsafe_allow_html=True)
                    for ev in evidences:
                        st.markdown(
                            f'<div class="ev"><div class="ev-t">{ev.get("title","未命名知识")}</div>'
                            f'<div class="ev-m">{ev.get("kbId","—")} ｜ '
                            f'{("更新 " + ev["updatedAt"]) if ev.get("updatedAt") else ("适用 " + (ev.get("scope") or "全店"))}'
                            f'{(" ｜ 风险 " + ev["risk"]) if ev.get("risk") else ""}</div>'
                            f'<div class="ev-x">{ev.get("excerpt","")}</div></div>',
                            unsafe_allow_html=True,
                        )

            with st.expander("建议生命周期", expanded=True):
                st.markdown(
                    f'<div class="state"><span class="dot" style="background:{LIFECYCLE_DOT.get(st_["lifecycle"], "#9ca3af")}"></span>'
                    f'当前状态：<b>{LIFECYCLE_LABEL.get(st_["lifecycle"], st_["lifecycle"])}</b></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="small">是否采纳：{"是" if st_["adopted"] else "否"}　'
                    f'是否改写：{"是" if st_["edited"] else "否"}</div>',
                    unsafe_allow_html=True,
                )
                if st_["final_text"]:
                    st.markdown('<div class="small">最终发送：</div>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="ev"><div class="ev-x">{st_["final_text"].replace("<","&lt;")}</div></div>',
                        unsafe_allow_html=True,
                    )
                if st_["outcome"]:
                    st.markdown(f'<div class="small">处理结果：{st_["outcome"]}</div>', unsafe_allow_html=True)

            with st.expander("操作审计（本次会话）", expanded=False):
                audit = st_["audit"]
                if not audit:
                    st.markdown('<div class="small">暂无操作记录。</div>', unsafe_allow_html=True)
                else:
                    for a in reversed(audit):
                        st.markdown(
                            f'<div class="small">{a["at"]}　<b>{a["action"]}</b>　{a["detail"]}</div>',
                            unsafe_allow_html=True,
                        )

            with st.expander("客户与订单", expanded=False):
                st.markdown(
                    f'<div class="small">姓名：{conv.get("customer",{}).get("name","—")}　'
                    f'等级：{conv.get("customer",{}).get("memberLevel","—")}<br>'
                    f'标签：{"、".join(conv.get("customer",{}).get("tags",[])) or "无"}<br>'
                    f'订单：{conv.get("orderId","—")}　渠道：{conv.get("channel","—")}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()
    if tab_live is not None:
        st.markdown(
            '<div class="small">想试「输入一句话，引擎现场回答」？切到上面的 '
            '<b>🤖 实时问答（引擎）</b> 页签——那里跑的是真引擎，不是剧本。</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="small">想换个问题验一验？用页面上方那个「试一句」输入框——'
            '它跑的是同一个 <code>policy.decide()</code>，不是剧本。</div>',
            unsafe_allow_html=True,
        )

# ================================================================ 实时问答
if tab_live is not None:
    with tab_live:
        st.markdown("### 输入一句客户问题，现场判定")
        st.markdown(
            '<div class="small">默认用 DeepSeek：它会读会话、翻 31 条知识库，再决定给建议、说不知道、还是转人工。'
            '右边的下拉可以切到「规则引擎」——那是不用模型的基线版本，同一句话两边一比，差别就看出来了。</div>',
            unsafe_allow_html=True,
        )
        trial_block("live", get_engine_kb(), load_eval_rows(), show_expected=True)

        st.divider()
        st.markdown("#### 切到「规则引擎」时，这几句会判成这样")
        st.markdown(
            '<div class="small">DeepSeek 模式下多数能判对；下面这些是**没有模型**时的真实表现，'
            '也就是 88% 里丢掉的那部分。</div>',
            unsafe_allow_html=True,
        )
        tb_rows = [
            ["退款三天了还没到账", "命中「退款多久到账」→ 正常给建议，并带出依据", "#059669"],
            ["我要退一赔三，不然就12315投诉", "命中风险词 → 高风险拦截，发送禁用", "#059669"],
            ["这个怎么样啊", "判错：给了护肤人群的答案，其实该转人工 —— 这就是 88% 剩下的 12%", "#dc2626"],
            ["你们能开发票吗", "判错得更明显：知识库没有发票规则，却命中「孕妇能用吗」并被判成高风险", "#dc2626"],
            ["拆封用过的美妆还能退吗", "同时命中互斥组 → 规则冲突：并列两条原文，勾选确认后才可发", "#d97706"],
        ]
        st.markdown(
            "".join(
                f'<div class="ev" style="border-left-color:{color}">'
                f'<div class="ev-x"><b>{q}</b> —— {note}</div></div>'
                for q, note, color in tb_rows
            ),
            unsafe_allow_html=True,
        )

def render_meta(tab_eval, tab_note) -> None:
    """评测与实现说明：只在 ?dev=1 时渲染。"""
    # ================================================================ 评测概览
    with tab_eval:
        rows = load_eval_rows()
        if not rows:
            st.warning(f"未找到评测文件：{EVAL_PATH}")
        else:
            versions = ["V1", "V2", "V3"]
            summary = {}
            for v in versions:
                rs = [r for r in rows if r["version"] == v]
                n = len(rs) or 1
                summary[v] = {
                    "n": len(rs),
                    "intent": sum(r["pred_intent"] == r["expected_intent"] for r in rs) / n,
                    "handoff": sum(truthy(r["pred_handoff"]) == truthy(r["expected_handoff"]) for r in rs) / n,
                    "risk": sum(r["pred_risk"] == r["expected_risk"] for r in rs) / n,
                    "key": sum(truthy(r["key_hit"]) for r in rs) / n,
                    "unsupported": sum(
                        1 for r in rs if r["answer"].strip() and not r["refs"].strip() and not truthy(r["pred_handoff"])
                    ),
                }

            st.markdown("### 评测总览")
            st.markdown(
                f'<div class="small">共 {summary["V3"]["n"]} 条评测用例 × 3 个版本，'
                f'规则版离线运行结果（n={summary["V3"]["n"]}），非估算。</div>',
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("评测用例", summary["V3"]["n"])
            c2.metric("通过（转人工判断正确）", f'{int(summary["V3"]["handoff"] * summary["V3"]["n"])}')
            c3.metric("失败", f'{summary["V3"]["n"] - int(summary["V3"]["handoff"] * summary["V3"]["n"])}')
            c4.metric("无依据回答", f'{summary["V3"]["unsupported"]} 条')

            st.markdown("### 三版对比")
            head = '<div class="card">'
            for v in versions:
                s = summary[v]
                head += (
                    f'<div style="margin-bottom:14px">'
                    f'<div class="card-h"><div class="card-t">{v}</div>'
                    f'<span class="chip blue">转人工 {s["handoff"]:.0%}</span></div>'
                    f'<div class="small">意图识别 {s["intent"]:.0%}　风险判定 {s["risk"]:.0%}　'
                    f'要点命中 {s["key"]:.0%}　无依据回答 {s["unsupported"]} 条</div>'
                    f'<div class="bar"><i style="width:{s["handoff"]*100:.0f}%"></i></div>'
                    f'</div>'
                )
            head += "</div>"
            st.markdown(head, unsafe_allow_html=True)

            st.markdown("### 失败用例（V3，转人工判断错误）")
            v3 = [r for r in rows if r["version"] == "V3"]
            fails = [r for r in v3 if truthy(r["pred_handoff"]) != truthy(r["expected_handoff"])]
            if fails:
                st.dataframe(
                    [
                        {
                            "用户问题": r["user_query"],
                            "期望转人工": "是" if truthy(r["expected_handoff"]) else "否",
                            "实际判定": "是" if truthy(r["pred_handoff"]) else "否",
                            "风险判定": f'{r["pred_risk"]} / 应为 {r["expected_risk"]}',
                        }
                        for r in fails
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success("V3 无转人工误判")

            st.markdown("### 单条试算")
            st.markdown(
                '<div class="small">把上表里任意一条问题粘进来，选 V3，就能看到引擎当场判错——'
                '这就是 88% 剩下的那 12%。换成 V2 或 V1，还能看到同一句在旧版本里判得更差。</div>',
                unsafe_allow_html=True,
            )
            trial_block(
            "eval", get_engine_kb(), rows, show_expected=True,
            modes=["规则 V3", "规则 V2", "规则 V1", "DeepSeek 生成"],
        )

            st.markdown("### 系统当前的能力边界")
            intent_err = [r for r in v3 if r["pred_intent"] != r["expected_intent"]]
            st.markdown(
                f"""
    <div class="card">
      <div class="card-body">· 意图识别停在 <b>82%</b>（{len(v3) - len(intent_err)}/{len(v3)}），三个版本完全持平——关键词法已到上限，继续加词不会提升。
    · 转人工判断 <b>88%</b>，剩余 {len(fails)} 条集中在两类：模糊问法（如「这个怎么样」）与越界问题（如「帮我写文案」）。
    · 无依据回答已为 <b>0</b>：没有知识就转人工，不再编答案。
    · 这是一份<b>历史基线</b>（规则引擎时代），与现在这套五态判定不是同一个系统；它只用来回答「为什么不再只追准确率」。</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

    # ================================================================ 实现说明
    with tab_note:
        st.markdown("### 当前实现边界")
        st.markdown(
            """
    - **知识库**：`data/knowledge_base.json`，31 条规则（售后 14 / 商品 9 / 物流 8），与评测用的是同一份；其中 2 条标了互斥组，用来产生「规则冲突」。
    - **判定层**：`policy.py`，一条线走完七步「识别意图 → 对话风险词 → 规则互斥冲突 → 查订单 → 本轮缺失字段 → 知识匹配 → 规则风险等级 → 出口」。界面上的建议卡、发送门禁、转人工原因读的都是同一个 `Verdict` 对象，不存在第二套判断；状态中文标签也只有一份（`policy.STATE_LABEL`），界面不再另写一套。
    - **检索**：`engine.py` 只负责召回（意图关键词 + 实词重合），把候选规则交给判定层，自己不下结论。
    - **会话数据**：`mock-data.json` 提供 7 条会话与消息。**每条会话的结论是打开时现场算的**，不是写死的：换成另一条会话、或在「试一句」里输入任意问题，结论会跟着变。
    - **DeepSeek**：只在 `ready` 状态下润色措辞，模型返回的 `state` 会被丢弃——判定权不在模型手里。没有 key 时用知识条目原文兜底，流程照样跑完。
    - **评测结果**：`data/evaluation_results.csv` 是 50 条用例 × 3 版本的历史结果，由规则引擎产出，**与现在这套五态不是同一个系统**，只作为「为什么不再只追准确率」的证据，不当作本版指标。
    - **状态**：发送、改写、忽略、转人工、审计记录存在 `st.session_state`，**刷新即重置**。
    - **没有的东西**：后端 API、SQLite/MySQL、登录鉴权、多客服协作、持久化审计。这些是正式系统的必需项，本原型故意不做——它们不服务于「证明 AI 会不会停下来」这个论点。
            """
        )

        st.markdown("### 预置的是数据，不是结论")
        st.markdown(
            """
    7 条会话、7 张订单、31 条规则都是编的，但**判定过程没有一条是编的**：结论由订单字段和规则原文算出来，所以它能自证——「缺退款单号」是因为 `refundNo` 是空的，「品质退货不可选」是因为订单里没有质量证据。

    面试里最容易被追问的一句是「如果换一个问题，屏幕上的结论会变吗」。演示时不要解释，直接把「试一句」打开，输入一句没准备过的话，让屏幕自己回答。
            """
        )

        st.markdown("### 面试时可以这么说")
        st.markdown(
            """
    > 当前版本是**前端交互原型**：用 JSON 提供初始数据，用 Session State 保存一次运行内的操作状态，没有后端和数据库。它要证明的是交互与判断逻辑，不是工程完备性。

    > 正式落地需要三件事：把知识与建议落库（MySQL），把每次建议的采纳/改写/发送/转人工写成审计事件，以及接入真实订单与物流的只读接口。这三件在「非目标」里写明过触发条件。
            """
        )

        st.markdown("### 建议生命周期是怎么记的")
        st.markdown(
            """
    | 状态 | 触发 | 对应问题 |
    | --- | --- | --- |
    | 未处理 pending | 系统生成建议，客服尚未动作 | AI 给了建议没人看？ |
    | 已采纳未发送 adopted | 点「采纳并填入」，但还没发送 | 采纳了又改吗？ |
    | 已编辑 edited | 发送内容与建议原文不一致 | 客服改了多少？ |
    | 已发送 sent | 原样发送 | 采纳率是多少？ |
    | 已忽略 ignored | 点「忽略建议」 | 哪些建议没用？ |
    | 已转人工 handed_off | 转人工并选原因 | 自动解决还是人工接管？ |
            """
        )

if show_meta:
    render_meta(tab_eval, tab_note)
