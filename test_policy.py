"""判定层验收：15 条用例，状态、门禁、归属必须同时成立。

跑法：python test_policy.py
任何一条 FAIL 就不算完成——因为界面上"卡片说不可发、按钮却能点"
这类矛盾，根源就是判定不唯一。
"""

from __future__ import annotations

import sys

from policy import compose_answer, decide, load_orders, STATE_LABEL

CASES = [
    # (说明, 问题, 订单号, 对话, 期望state, 期望可发, 期望缺失字段, 期望证据前缀)
    ("C1 退款三天未到账（退款单未生成）", "退款申请提交三天了怎么还没到账",
     "SO20261105001", "你好，我上周买的那个精华，退款申请提交三天了，怎么还没到账？",
     "need_info", False, "退款单号", "售后规则-004"),
    ("C2 使用后过敏（健康风险）", "用了两天脸过敏了怎么办",
     "SO20261104037", "你们这个质量也太差了吧，用了两天就烂脸了",
     "blocked", False, None, None),
    ("C3 物流停在沈阳三天", "我的快递停在沈阳三天了怎么回事",
     "SO20261102099", "我的快递停在沈阳三天了怎么回事",
     "ready", True, None, "物流规则"),
    ("C5 价保：签收第 6 天降价（缺活动标记）", "我前天买的，今天降了 40 块，能补差价吗？",
     "SO20261029012", "我前天买的，今天降了 40 块，能补差价吗？",
     "need_info", False, "价保适用性", "售后规则-014"),
    ("C7 发票（知识库无此规则）", "你们能开发票吗",
     "", "你们能开发票吗",
     "no_knowledge", False, None, None),
    ("查无此单（客户报错订单号）", "我的快递到哪了",
     "SO20260000000", "我的快递到哪了",
     "need_info", False, "订单号", None),
    ("售后问题但会话无订单号", "我要退货怎么退",
     "", "我要退货怎么退",
     "need_info", False, "订单号", None),
    ("B 开封后能用多久（参数为空）", "这个开封后能用多久",
     "SO20261106018", "这个面霜开封后能用多久",
     "need_info", False, "开封后使用期限", None),
    ("B 同一张单问退款（缺失字段与本次无关）", "退款到账了吗",
     "SO20261106018", "退款到账了吗",
     "ready", True, None, "售后规则"),
    ("拆封用过能否退（两条规则互斥）", "拆封用过的美妆还能退吗",
     "", "拆封用过的美妆还能退吗",
     "conflict", True, None, "售后规则"),
    ("上一句提过 12315，这一句没提", "你们到底什么时候给答复",
     "", "我要退一赔三，不然就去12315投诉。你们到底什么时候给答复",
     "blocked", False, None, None),
    ("C6 换货单未落库（不能报出库时间）", "我上周寄回去换的货，什么时候能发出来？",
     "SO20261031055", "我上周寄回去换的货，什么时候能发出来？",
     "need_info", False, "换货单号", None),
    ("C8 买错色号想换（同义说法必须召回）", "买错色号了想换一个",
     "SO20261106012", "买错色号了想换一个",
     "ready", True, None, "售后规则-005"),
    ("规则高风险：显示签收但没收到", "显示签收但我没收到",
     "SO20261031055", "显示签收但我没收到",
     "blocked", False, None, "物流规则-008"),
]


def run() -> int:
    orders_doc = load_orders()
    passed = failed = 0

    for (desc, question, order_no, dialog, want_state, want_send,
         want_missing, want_prefix) in CASES:
        v = decide(question, order_no=order_no, orders_doc=orders_doc, dialog=dialog)
        problems = []

        if v.state != want_state:
            problems.append(f"state={v.state} 期望 {want_state}")
        if v.can_send != want_send:
            problems.append(f"can_send={v.can_send} 期望 {want_send}")
        if want_missing:
            fields = "、".join(m.field for m in v.missing)
            if want_missing not in fields:
                problems.append(f"missing={fields or '空'} 期望含 {want_missing}")
        if want_prefix:
            ids = [e["kbId"] for e in v.evidence]
            if not any(i.startswith(want_prefix) for i in ids):
                problems.append(f"evidence={ids} 期望命中 {want_prefix}*")

        # 门禁不变式：不可发的状态必须一定不可发，且必须有归属人
        if not v.can_send and v.state in {"ready"}:
            problems.append("ready 却不可发")
        if v.state == "blocked" and v.handoff_reason != "high_risk":
            problems.append(f"blocked 的转人工原因={v.handoff_reason}")
        if v.state == "conflict" and not v.requires_ack:
            problems.append("conflict 未要求确认")
        if not v.owner:
            problems.append("没有归属人")

        # 判定步骤必须真实存在，且从识别意图开始——界面渲染的是它，不是另写的模板
        if not v.steps or v.steps[0][0] != "识别意图":
            problems.append("判定步骤缺失或顺序不对")

        # 冲突态不能把 high 风险规则放出去（高风险必须在冲突之前或同层被拦住）
        if v.state == "conflict" and any(e.get("risk") == "high" for e in v.evidence):
            problems.append("conflict 里含 high 风险规则")

        # ready 的物流回答必须用订单运单号，且不能声称已经执行了没有记录的动作
        if v.state == "ready" and v.intent == "物流咨询":
            text, src = compose_answer(v)
            tail = ((v.order or {}).get("logistics") or {}).get("waybill", "")[-4:]
            if src != "order" or (tail and tail not in text):
                problems.append("物流回答没有用订单运单号")
            if "已为您登记" in text:
                problems.append("物流回答声称已完成未执行的动作")

        flag = "PASS" if not problems else "FAIL"
        if problems:
            failed += 1
        else:
            passed += 1
        print(f"[{flag}] {desc}")
        print(f"       {STATE_LABEL[v.state]}｜可发={v.can_send}｜归属={v.owner}")
        print(f"       为什么：{v.why}")
        if v.missing:
            print(f"       缺：{'；'.join(m.field for m in v.missing)}")
        if v.evidence:
            print(f"       依据：{v.evidence[0]['kbId']} {v.evidence[0]['title'][:24]}")
        if problems:
            print(f"       !!! {' / '.join(problems)}")
        print()

    # ---- 第 15 条：客户口径 vs 字段口径 ----
    # 这是本项目主张「只说有依据的话」最直接的一条验收：
    # 客户说"五天没动"，运单节点时间算出来是 4 天，话术必须用 4 天，
    # 并且把这次对账记进判定步骤，让客服看得见系统纠正了客户口径。
    q15 = "我的快递五天没动了，一直显示在转运中心"
    v = decide(q15, order_no="SO20261102099", orders_doc=orders_doc,
               dialog=q15, as_of="2026-11-05T10:35:00+08:00")
    text, src = compose_answer(v)
    problems = []
    if v.state != "ready":
        problems.append(f"state={v.state} 期望 ready")
    if "已经 4 天没有更新" not in text:
        problems.append(f"话术未按节点时间算天数：{text}")
    if "5 天" in text:
        problems.append("话术抄了客户口径的 5 天")
    if not any(name == "核对客户口径" for name, _ in v.steps):
        problems.append("没有记录客户口径与字段口径的对账步骤")
    if "已为您登记" in text:
        problems.append("声称已完成未执行的动作")
    flag = "PASS" if not problems else "FAIL"
    if problems:
        failed += 1
    else:
        passed += 1
    print(f"[{flag}] 客户说 5 天，字段算 4 天（话术必须用字段口径）")
    print(f"       {STATE_LABEL[v.state]}｜可发={v.can_send}｜归属={v.owner}")
    print(f"       话术：{text}")
    if problems:
        print(f"       !!! {' / '.join(problems)}")
    print()

    total = len(CASES) + 1
    print(f"结果：{passed} 通过 / {failed} 失败，共 {total} 条")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
