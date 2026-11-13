"""统一判定层：全项目唯一的判定入口。

为什么要单独一层
----------------
之前建议卡自己猜状态、发送按钮自己再猜一次，两边各写一套 if，
于是出现"卡片说不可发、按钮却能点"这类自相矛盾。面试官只要问一句
"它凭什么禁用"，就会暴露成两套逻辑。这里把判定收成一条线：

    decide(问题, 订单号) -> Verdict

建议卡、发送门禁、转人工原因、审计日志全部只读 Verdict，不再各自判断。

判定顺序（顺序本身就是产品主张）
--------------------------------
1. 风险词（只看对话，不需要订单）——投诉、法律、舆情、人身安全
2. 规则冲突（只依赖规则，也不需要订单）
3. 订单是否查得到（只有"这个问题需要订单事实"时才查）
4. 缺字段（只缺"这次真要用到"的字段）
5. 知识匹配（有没有可引用的规则）
6. 规则风险等级（命中的规则标了 risk_level=high，也不自动作答）
7. 出口

第 1、2 步放在查订单之前，是因为它们不依赖订单事实。若先查订单，
"规则冲突但订单号打错"时会先报 need_info，把冲突藏起来——藏起来的
冲突就是下一次赔付。

越靠前越"不可协商"：风险优先于冲突，冲突优先于政策，政策优先于知识检索。
LLM 不参与判定，它只负责把 Verdict 写成话术（见 llm.generate_suggestion）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine import (
    HANDOFF_KEYWORDS,
    detect_intent,
    find_conflict,
    load_kb,
    retrieve_hits,
)

BASE_DIR = Path(__file__).resolve().parent
ORDERS_PATH = BASE_DIR / "data" / "orders.json"

# 状态枚举：中文名与"这条状态下系统允许做什么"一一对应
STATE_LABEL = {
    "ready": "可发建议",
    "need_info": "信息不足",
    "conflict": "规则冲突",
    "blocked": "高风险拦截",
    "no_knowledge": "知识不足",
    "error": "查询失败",
}

# 谁能接手这件事——不是"谁都能管"，必须指名
OWNER_LABEL = {
    "ready": "客服",
    "need_info": "客服",
    "conflict": "客服（判断不了升级组长）",
    "blocked": "客服组长",
    "no_knowledge": "知识库管理员",
    "error": "客服（保留人工入口）",
}

# 缺字段只有在"客户这次真的问到它"时才拦。否则退货单缺个色号，
# 客户问物流也会被误判成信息不足。
FIELD_TRIGGERS = {
    "开封后使用期限": ["开封", "过期", "保质", "期限", "能用多久", "放多久"],
    "退款单号": ["到账", "退款", "退钱"],
    "色号": ["色号", "颜色", "哪个色", "显白"],
    "容量": ["容量", "多少毫升", "多少克", "规格"],
    "换货单号": ["换货", "寄回", "什么时候能发", "什么时间能发", "发出", "寄出"],
    "价保适用性": ["价保", "补差价", "降价", "差价"],
}

# 这些问题必须依赖订单事实，没有订单号就不能往下判
ORDER_REQUIRED_INTENTS = {"物流咨询", "售后咨询"}

# 字符重合检索的噪声来源：功能词两两组合也能"命中"。
# 例：「你们能开发票吗」会命中「孕妇能用吗」，共用的是"能/用/吗"，不是业务。
# 所以再加一道相关性检查：至少共享一个**实词二元组**。
STOP_BIGRAMS = {
    "你们", "我们", "他们", "我的", "你的", "这个", "那个", "这款", "那款",
    "怎么", "什么", "为什", "么样", "多少", "可以", "是不", "不是", "的话",
    "了吗", "呢吗", "能吗", "有吗", "能不", "请问", "一下", "就是", "还是",
    "签收", "收到",  # 「签收/收到」几乎每条售后规则里都有，用它命中等于没命中
}


@dataclass
class Missing:
    field: str
    ask: str


@dataclass
class Verdict:
    """一次判定的完整结果。UI 只能读这个对象，不允许再自己推。"""

    state: str
    why: str
    question: str
    intent: str
    owner: str = ""
    order_no: str = ""
    # 判定的参考时刻（模拟数据没有真实时钟，取会话最后一条消息的时间）。
    # 话术里所有"停了几天"都必须由它减订单字段算出来，不许抄客户原话。
    as_of: str = ""
    order: dict | None = None
    evidence: list[dict] = field(default_factory=list)
    missing: list[Missing] = field(default_factory=list)
    risks: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    # 判定过程：每走一步记一条 (做了什么, 结果)。UI 只渲染，不允许反推。
    steps: list[tuple[str, str]] = field(default_factory=list)

    def step(self, name: str, result: str) -> "Verdict":
        self.steps.append((name, result))
        return self

    # ---- 派生字段：门禁与文案都从这里取，保证只有一套规则 ----
    @property
    def can_send(self) -> bool:
        """能不能点发送。只有 ready 与 conflict（需确认）可以。"""
        if self.state == "ready":
            return True
        if self.state == "conflict":
            return True
        return False

    @property
    def requires_ack(self) -> bool:
        """conflict 允许发，但必须客服先确认"两条规则我都看过"。"""
        return self.state == "conflict"

    @property
    def handoff_reason(self) -> str:
        """建议的转人工原因码，与 handoffReasons 对齐。"""
        return {
            "blocked": "high_risk",
            "conflict": "conflict",
            "no_knowledge": "no_knowledge",
            "error": "other",
        }.get(self.state, "")

    @property
    def badge(self) -> str:
        return STATE_LABEL.get(self.state, self.state)

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "why": self.why,
            "owner": self.owner,
            "order_no": self.order_no,
            "evidence": [e.get("kbId") for e in self.evidence],
            "missing": [m.field for m in self.missing],
            "can_send": self.can_send,
            "requires_ack": self.requires_ack,
            "handoff_reason": self.handoff_reason,
        }


# ----------------------------------------------------------------- 数据访问


def load_orders(path: Path | str = ORDERS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lookup_order(order_no: str, orders_doc: dict | None = None) -> tuple[str, dict | None]:
    """三出口：found / not_found / no_number。

    "查无此单"和"没给单号"必须分开——前者是客户说错了，
    后者是我们还没问。混成一个分支，话术一定串。
    """
    doc = orders_doc if orders_doc is not None else load_orders()
    order_no = (order_no or "").strip()
    if not order_no:
        return "no_number", None
    for item in doc.get("orders", []):
        if item.get("orderId") == order_no:
            return "found", item
    return "not_found", None


# ----------------------------------------------------------------- 各段判定


def _risk_hits(text: str) -> list[dict]:
    hits = []
    for word in HANDOFF_KEYWORDS:
        if word in text:
            level = "high" if word in {"12315", "投诉", "曝光", "律师", "消协", "过敏", "烂脸", "起痘", "刺痛", "赔偿", "赔钱"} else "medium"
            hits.append({"level": level, "reason": f"对话命中「{word}」"})
    return hits


def _bigrams(text: str) -> set[str]:
    keep = "".join(ch for ch in (text or "") if ch.isalnum())
    return {keep[i:i + 2] for i in range(len(keep) - 1)} - STOP_BIGRAMS


def _relevance(item: dict, question: str) -> int:
    """问题与规则标题/条件的实词二元组重合数。

    只比标题与适用条件，不比 answer——比 answer 会让模板话术（“我们为您登记并转专员”）
    跟任何售后问题都撞上，把不相关的规则也捞进来。
    """
    qb = _bigrams(question)
    ib = _bigrams(item.get("question", "") + item.get("conditions", ""))
    return len(qb & ib)


def _trigger_hit(item: dict, question: str) -> bool:
    """规则自带的触发短语是否出现在问题里。

    汉字同义不同形（客户说“没动”，规则写“不动”），二元组对不上，
    所以每条规则可以带一组触发短语，命中任意一个就算这条规则相关。
    """
    for t in item.get("triggers") or []:
        if t and t in question:
            return True
    return False


def _recall(question: str, kb: list[dict], intent: str) -> list[tuple[int, dict]]:
    """召回 = 字符重合 + 触发短语。

    只做字符重合时，客户换个说法（“买错色号想换一个”）根本进不了候选池，
    触发器再准也只能过滤、不能补招。这里把触发短语也当成一条召回通道；
    类目与本次意图不一致的规则在召回阶段就丢掉，避免隔壁场景被捞进来。
    """
    hits = retrieve_hits(question, kb)
    seen = {id(item) for _score, item in hits}
    for item in kb:
        if id(item) in seen:
            continue
        if item.get("category") and intent and item["category"] != intent:
            continue
        if _trigger_hit(item, question):
            hits.append((1, item))
            seen.add(id(item))
    return hits


def _filter_hits(hits: list[tuple[int, dict]], question: str, intent: str) -> list[tuple[int, dict]]:
    """在字符重合之后再加一道相关性闸门，并按相关性重排。

    三道条件同时成立才算命中：
    ① 规则类目与本次意图一致（挡住"问退款却调出查快递"）；
    ② 触发短语命中，或至少共享 2 个实词二元组
       （挡住「快递显示不动」撞「显示签收但没收到」这种同词不同事）；
    ③ 同分时再按原始字符重合度排。
    任一条不满足就丢弃——宁可判 no_knowledge，也不拿隔壁场景的规则凑答案。
    """
    kept = []
    for score, item in hits:
        if item.get("category") and intent and item["category"] != intent:
            continue
        rel = _relevance(item, question)
        if not (_trigger_hit(item, question) or rel >= 2):
            continue
        kept.append((rel, item, score))
    kept.sort(key=lambda t: (t[0], t[2]), reverse=True)
    return [(score, item) for _rel, item, score in kept]


def _needs_order(intent: str, question: str) -> bool:
    if intent in ORDER_REQUIRED_INTENTS:
        return True
    return any(w in question for words in FIELD_TRIGGERS.values() for w in words)


def _missing_for(order: dict | None, question: str) -> list[Missing]:
    if not order:
        return []
    out = []
    for name in order.get("missingFields") or []:
        triggers = FIELD_TRIGGERS.get(name)
        if triggers is None or any(w in question for w in triggers):
            out.append(
                Missing(
                    field=name,
                    ask=_ask_text(order, name),
                )
            )
    return out


def _ask_text(order: dict, name: str) -> str:
    if name == "退款单号":
        return f"这笔 {order.get('orderId')} 的退款单没有生成，不能按“在途”解释；需核对该笔退款是否落库，或引导客户重新提交。"
    if name == "开封后使用期限":
        return f"商品参数里「开封后使用期限」为空，不能报数字；需向仓库/商品管理员确认后再答。"
    if name == "色号":
        return f"订单未记录色号，先确认客户要换的颜色，再判断能不能换。"
    if name == "容量":
        return f"商品参数里「容量」为空，不能报数字；需补齐参数后再答。"
    if name == "换货单号":
        return f"这笔 {order.get('orderId')} 的换货单没有落库，无法报出库时间；需核对该换货单是否生成，不能按“在仓库处理中”口头承诺。"
    if name == "价保适用性":
        return f"这笔 {order.get('orderId')} 没有记录是否参加限时秒杀/优惠券叠加，不能直接承诺价保；需先核对活动标记（售后规则-014）。"
    return f"「{name}」缺失，需要先补齐。"


def _evidence(hits: list[tuple[int, dict]]) -> list[dict]:
    out = []
    for score, item in hits[:2]:
        out.append(
            {
                "kbId": item.get("id", ""),
                "title": item.get("question", ""),
                "excerpt": item.get("answer", ""),
                "scope": item.get("conditions", ""),
                "risk": item.get("risk_level", ""),
                "updatedAt": item.get("updated_at", ""),
                "score": score,
            }
        )
    return out


# ----------------------------------------------------------------- 主入口


def decide(
    question: str,
    order_no: str = "",
    kb: list[dict] | None = None,
    orders_doc: dict | None = None,
    dialog: str = "",
    as_of: str = "",
) -> Verdict:
    """唯一判定入口。每走一步都往 v.steps 里记一条，界面只渲染不推断。"""
    question = (question or "").strip()
    dialog = (dialog or question).strip()
    kb = kb if kb is not None else load_kb()

    intent = detect_intent(question)
    raw_hits = _recall(question, kb, intent)
    hits = _filter_hits(raw_hits, question, intent)

    v = Verdict(
        state="", why="", question=question, intent=intent,
        order_no=order_no, as_of=(as_of or "").strip(),
    )
    v.step("识别意图", intent or "无法归类")
    v.step("检索规则", f"粗召回 {len(raw_hits)} 条 → 相关性筛后 {len(hits)} 条")

    # ① 风险：只看对话，不看订单。投诉/法律/舆情/健康风险一旦出现，
    #    不该再去检索政策——检索得越准，越可能把"承诺"说得更像真的。
    risks = _risk_hits(dialog)
    if risks:
        v.step("检查风险词", f"命中 {len(risks)} 个：{'、'.join(r['reason'] for r in risks[:3])}")
        v.state = "blocked"
        v.why = "对话里出现投诉、法律、舆情或健康风险，系统不生成可发送内容。"
        v.owner = OWNER_LABEL["blocked"]
        v.risks = risks
        v.evidence = _evidence(hits)
        v.actions = ["转人工（组长）", "保留上下文与命中规则，供事后复盘"]
        return v
    v.step("检查风险词", "未命中")

    order = None
    # ② 冲突：只依赖规则，不依赖订单。
    conflict_items = find_conflict(hits, question)
    if conflict_items:
        v.step("检查规则冲突", f"命中 {len(conflict_items)} 条互斥规则")
        v.state = "conflict"
        v.why = f"同时命中 {len(conflict_items)} 条结论相反的规则，系统不合并、不代选。"
        v.owner = OWNER_LABEL["conflict"]
        v.evidence = _evidence(hits)
        v.notes = [f"{i.get('id')}：{i.get('answer')}" for i in conflict_items]
        v.actions = ["并列展示两条规则原文", "客服确认后再发送；判断不了升级组长"]
        return v
    v.step("检查规则冲突", "无互斥")

    # ③ 订单：只有这个问题需要订单事实时才查
    if _needs_order(intent, question):
        status, order = lookup_order(order_no, orders_doc)
        if status == "not_found":
            v.step("查订单", f"{order_no} 查无此单")
            v.state = "need_info"
            v.why = f"订单号 {order_no} 查不到，不能套用其他订单的政策。"
            v.owner = OWNER_LABEL["need_info"]
            v.missing = [Missing(field="订单号", ask="请客户确认订单号，或核对是否下错店铺。")]
            v.actions = ["向客户确认订单号"]
            return v
        if status == "no_number":
            v.step("查订单", "没有订单号")
            v.state = "need_info"
            v.why = "这个问题需要订单事实，但当前会话没有可用的订单号。"
            v.owner = OWNER_LABEL["need_info"]
            v.missing = [Missing(field="订单号", ask="请客户提供订单号。")]
            v.actions = ["向客户索取订单号"]
            return v
        v.step("查订单", f"{order_no} → 命中")
    else:
        v.step("查订单", "这个问题不需要订单字段，未查单")

    # ④ 缺字段：只拦"这次真要用到"的字段
    missing = _missing_for(order, question)
    if missing:
        names = "、".join(m.field for m in missing)
        v.step("查字段", f"「{names}」为空")
        v.state = "need_info"
        v.why = f"要用到的字段「{names}」在订单里是空的，不能给它编一个值。"
        v.owner = OWNER_LABEL["need_info"]
        v.order = order
        v.missing = missing
        v.evidence = _evidence(hits)[:1]
        v.actions = [m.ask for m in missing]
        return v
    v.step("查字段", "本次要用的字段齐全" if order else "无订单字段需求")

    # ⑤ 知识：有没有能引用的规则
    if not hits:
        v.step("匹配知识", "0 条可用规则")
        v.state = "no_knowledge"
        v.why = "知识库里没有能支撑回答的规则，系统不说“大概是这样”。"
        v.owner = OWNER_LABEL["no_knowledge"]
        v.order = order
        v.actions = ["转人工", "登记知识缺口，由知识管理员补规则"]
        return v
    v.step("匹配知识", f"采用 {hits[0][1].get('id')}（相关度 {_relevance(hits[0][1], question)}）")

    # ⑥ 高风险规则：risk_level=high 表示"这条不能自动作答"，一律交人工
    high = [item for _s, item in hits if item.get("risk_level") == "high"]
    if high:
        ids = "、".join(i.get("id", "") for i in high)
        v.step("检查规则风险等级", f"{ids} 为 high，不允许自动作答")
        v.state = "blocked"
        v.why = f"命中的规则「{ids}」风险等级为高，系统不自动生成答复。"
        v.owner = OWNER_LABEL["blocked"]
        v.order = order
        v.evidence = _evidence(hits)
        v.actions = ["转人工（组长）", "由专人按规则处理并留痕"]
        return v
    v.step("检查规则风险等级", "非高风险")

    # ⑦ 出口：可发，每句都要有出处
    v.state = "ready"
    v.why = f"命中规则 {hits[0][1].get('id')}，订单字段齐全，可以给建议。"
    v.owner = OWNER_LABEL["ready"]
    v.order = order
    v.evidence = _evidence(hits)
    v.actions = ["采纳或改写后发送", "发送后记录本条建议是否被采用"]
    v.step("出口", "可发建议")
    return v


_CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _days_mentioned(text: str) -> int | None:
    """取客户原话里的"几天"。只用于和字段口径对账，不作为话术依据。"""
    import re

    m = re.search(r"([0-9]+|[一二两三四五六七八九十])\s*天", text or "")
    if not m:
        return None
    tok = m.group(1)
    return int(tok) if tok.isdigit() else _CN_NUM.get(tok)


def compose_answer(v: Verdict) -> tuple[str, str]:
    """用订单字段拼出可发的这句话，而不是把知识库原文当答复。

    返回 (话术, 来源)：来源 "order" = 用订单字段拼的；"kb" = 只能引用规则原文。
    """
    order = v.order or {}
    ev = v.evidence[0] if v.evidence else {}
    if v.intent == "物流咨询" and order.get("logistics"):
        lg = order["logistics"]
        tail = (lg.get("waybill") or "")[-4:]
        raw_node = (lg.get("lastNode") or "").strip()
        # 平台字段形如「快件【沈阳转运中心】」，只取方括号里的地名
        m = re.search(r"【(.+?)】", raw_node)
        node = m.group(1) if m else raw_node.replace("快件", "").strip("，, ")
        days = _days_between(v.as_of, lg.get("lastNodeAt"))
        claimed = _days_mentioned(v.question)
        where = f"在{node}" if node else "在途中"
        # 天数只能由「参考时刻 − 物流节点时间」算出来。客户说"五天没动"是他的口径，
        # 不是依据——这里曾经直接抄客户那个数字，等于系统自己违反"只说有依据的话"。
        if days is not None:
            lead = f"您反馈运单尾号 {tail} 的包裹{where}已经 {days} 天没有更新，属于异常。"
            if claimed is not None and claimed != days:
                v.step(
                    "核对客户口径",
                    f"客户说 {claimed} 天，按节点时间算 {days} 天，话术采用字段口径",
                )
        else:
            lead = (
                f"运单尾号 {tail} 的包裹目前{where}，节点更新时间缺失，"
                "算不出停滞天数，先按异常登记催件。"
            )
        return (
            f"{lead}按规则可以为您登记催件并转专员跟进；需要我现在提交登记吗？",
            "order",
        )
    text = ev.get("excerpt") or ""
    return (text, "kb")


def order_summary(order: dict | None) -> dict:
    """给界面用的订单摘要：只暴露判定用得到的字段。"""
    if not order:
        return {}
    return {
        "orderId": order.get("orderId", ""),
        "title": order.get("title", ""),
        "customer": order.get("customer", ""),
        "amount": order.get("amount", 0),
        "spec": order.get("spec", {}),
        "logistics": order.get("logistics", {}),
        "afterSale": order.get("afterSale", {}),
        "refund": order.get("refund", {}),
        "missingFields": order.get("missingFields", []),
    }


# ---------------------------------------------------------------------------
# A 场景：退货原因的选择后果
#
# 这一屏只回答一个问题：同一个诉求，选哪个原因，商家各要掏多少钱、被记几分。
# 三个原因都"走得通"，但只有一个是把它记成商责。客服为了省事选品质退货，
# 客户不用付运费、退款更快，代价落在商家账上——这就是评论里"这月帮我
# 品退大几十单"的机制。
# ---------------------------------------------------------------------------

ASSUMED_SHIPPING = 12  # 单程运费假设（元），讲的时候要说"假设，不是实测"

REASON_LABEL = {
    "seven_day": "七天无理由退货",
    "exchange": "换货（同款换色号）",
    "quality_return": "品质退货",
}

_QUALITY_WORDS = ("质量", "破损", "瑕疵", "漏液", "变质", "过敏", "异味")


@dataclass
class ReasonOption:
    code: str
    label: str
    allowed: bool
    why: str
    seller_cost: int
    customer_pays_return: bool
    counts_return_rate: bool
    affects_score: bool

    @property
    def cost_text(self) -> str:
        if self.seller_cost <= 0:
            return "商家不额外掏（寄回运费由客户承担）"
        return f"商家多掏约 ¥{self.seller_cost}"


def _days_between(later: str | None, earlier: str | None) -> int | None:
    from datetime import datetime

    if not later or not earlier:
        return None
    try:
        a = datetime.fromisoformat(later)
        b = datetime.fromisoformat(earlier)
    except ValueError:
        return None
    return max((a - b).days, 0)


def _day_label(days: int) -> str:
    return "签收当天" if days == 0 else f"签收后第 {days} 天"


def return_reasons(order: dict) -> list[ReasonOption]:
    """给出三个退货原因各自"允不允许 + 代价"。

    判定依据全部来自订单字段与客户原话，不猜：
    - 是否拆封/使用 → 决定七天无理由能不能用（美妆个护一经拆封不支持）
    - 距签收天数 → 决定七天无理由（7 天）与换货（15 天）是否在窗口内
    - 客户原话里有没有质量类词 → 决定品质退货能不能选
    """
    after = (order or {}).get("afterSale") or {}
    opened = bool(after.get("opened"))
    used = bool(after.get("used"))
    reason_text = f"{after.get('reason') or ''}"
    days = _days_between(after.get("appliedAt"), (order or {}).get("deliveredAt"))
    has_quality_evidence = any(w in reason_text for w in _QUALITY_WORDS)

    out: list[ReasonOption] = []

    # ① 七天无理由
    if opened:
        why_7d = "已拆封：美妆个护一经拆封不支持七天无理由（售后规则-001 的适用条件与不可答说明）。"
        ok_7d = False
    elif days is None:
        why_7d = "缺签收时间，算不出是否在 7 天内，不能承诺可退。"
        ok_7d = False
    elif days > 7:
        why_7d = f"{_day_label(days)}，已超出 7 天窗口。"
        ok_7d = False
    else:
        why_7d = f"未拆封、{_day_label(days)}，符合七天无理由。"
        ok_7d = True
    out.append(
        ReasonOption(
            code="seven_day",
            label=REASON_LABEL["seven_day"],
            allowed=ok_7d,
            why=why_7d,
            seller_cost=0,
            customer_pays_return=True,
            counts_return_rate=False,
            affects_score=False,
        )
    )

    # ② 换货
    if days is None:
        ok_ex, why_ex = False, "缺签收时间，算不出换货窗口。"
    elif days > 15:
        ok_ex, why_ex = False, f"{_day_label(days)}，已超出 15 天换货窗口（售后规则-005）。"
    else:
        ok_ex, why_ex = True, f"{_day_label(days)}，在 15 天换货窗口内（售后规则-005）；客户诉求是换不是退。"
    out.append(
        ReasonOption(
            code="exchange",
            label=REASON_LABEL["exchange"],
            allowed=ok_ex,
            why=why_ex,
            seller_cost=ASSUMED_SHIPPING,
            customer_pays_return=True,
            counts_return_rate=False,
            affects_score=False,
        )
    )

    # ③ 品质退货
    if has_quality_evidence:
        ok_q, why_q = True, "客户原话提到质量问题，但需医院证明或质检报告后才能按质量处理（订单 policyHooks K2；过敏类另见售后规则-003，必须转人工）。"
    else:
        ok_q = False
        why_q = (
            "订单里没有任何质量证据：未拆封、未使用，客户原话是"
            f"「{reason_text or '未填写'}」。选它等于把非质量问题记成品退。"
        )
    out.append(
        ReasonOption(
            code="quality_return",
            label=REASON_LABEL["quality_return"],
            allowed=ok_q,
            why=why_q,
            seller_cost=ASSUMED_SHIPPING * 2,
            customer_pays_return=False,
            counts_return_rate=True,
            affects_score=True,
        )
    )
    return out


if __name__ == "__main__":  # 手工试一条
    import sys

    q = " ".join(sys.argv[1:]) or "退款申请提交三天了怎么还没到账"
    v = decide(q, order_no="SO20261105001")
    print(json.dumps(v.as_dict(), ensure_ascii=False, indent=2))
    print(v.why)
