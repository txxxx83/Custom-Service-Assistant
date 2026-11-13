# -*- coding: utf-8 -*-
"""核对手册里所有"可现场复现"的断言。跑法：python _scratch/verify_claims.py

这是文档的唯一复算入口：任何文档里的数字/状态/规则 ID 都应该能在这里跑出来。
跑完看最后一行「断言汇总」，必须 0 失败。
"""
import sys, json, io, csv, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"D:\Desktop\产品\demo"
sys.path.insert(0, ROOT)
import policy, engine

kb = engine.load_kb()
od = json.load(open(os.path.join(ROOT, "data", "orders.json"), encoding="utf-8"))
md = json.load(open(os.path.join(ROOT, "mock-data.json"), encoding="utf-8"))
ORDERS = od["orders"]  # list，主键 orderId

FAIL = []
PASS = 0


def check(label, got, want):
    global PASS
    if got == want:
        PASS += 1
        print(f"  [OK]   {label}: {got}")
    else:
        FAIL.append(label)
        print(f"  [FAIL] {label}: 实际={got!r} 期望={want!r}")


def conv_as_of(cid):
    """与 app.conv_as_of 同口径：该会话最后一条消息的时间。"""
    return max((m.get("at", "") for m in md["messages"] if m.get("convId") == cid), default="")


def customer_dialog(cid):
    """与 app 同口径：整段客户发言拼接，不是只取最后一句。"""
    return "\n".join(m["text"] for m in md["messages"]
                     if m.get("convId") == cid and m.get("from") == "customer")


def run(q, order_no="", as_of=""):
    v = policy.decide(q, order_no=order_no, kb=kb, orders_doc=od, dialog=q, as_of=as_of)
    ev = [e.get("kbId") for e in (v.evidence or [])]   # 键是 kbId，不是 ruleId/id
    print(f"  {q!r} order={order_no or '(无)'} -> {v.state} | can_send={v.can_send} "
          f"| ev={ev} | missing={[m.field for m in (v.missing or [])]}")
    return v


print("=== 一、7 条会话实测判定（用 app.py 同一调用路径复算）===")
EXPECT = {
    "C1": ("need_info", "售后规则-004", "退款单号", "客服"),
    "C2": ("blocked", "售后规则-012", None, "客服组长"),
    "C3": ("ready", "物流规则-003", None, "客服"),
    "C4": ("blocked", "售后规则-005", None, "客服组长"),
    "C5": ("need_info", "售后规则-014", "价保适用性", "客服"),
    "C6": ("need_info", "售后规则-005", "换货单号", "客服"),
    "C7": ("no_knowledge", None, None, "知识库管理员"),
}
CONV_BY_ID = {c["id"]: c for c in md["conversations"]}
dist = {}
for cid, (st, rule, miss, owner) in EXPECT.items():
    conv = CONV_BY_ID[cid]
    q = customer_dialog(cid)
    v = policy.decide(q, order_no=conv.get("orderId", ""), kb=kb, orders_doc=od,
                      dialog=q, as_of=conv_as_of(cid))
    ev = [e.get("kbId") for e in (v.evidence or [])]
    mf = [m.field for m in (v.missing or [])]
    print(f"  {cid} {conv.get('customer','')} -> {v.state} | ev={ev} | missing={mf} | owner={v.owner}")
    check(f"{cid} state", v.state, st)
    check(f"{cid} 首条依据", ev[0] if ev else None, rule)
    check(f"{cid} 缺失字段", miss in mf if miss else not mf, True)
    check(f"{cid} 归属", v.owner, owner)
    dist[v.state] = dist.get(v.state, 0) + 1
print("  分布:", dist)
check("分布 1可发/3信息不足/2拦截/1知识不足",
      (dist.get("ready", 0), dist.get("need_info", 0), dist.get("blocked", 0), dist.get("no_knowledge", 0)),
      (1, 3, 2, 1))

print("\n=== 二、C3 客户口径 vs 字段口径（第 15 条用例守的纪律）===")
q3 = customer_dialog("C3")
v3 = policy.decide(q3, order_no="SO20261102099", kb=kb, orders_doc=od,
                   dialog=q3, as_of=conv_as_of("C3"))
text3, _ = policy.compose_answer(v3)
print("  话术:", text3)
check("C3 as_of = 会话末条消息时间", v3.as_of, conv_as_of("C3"))
check("C3 话术含 4 天", "已经 4 天没有更新" in text3, True)
check("C3 话术不抄客户的 5 天", "5 天" in text3, False)
check("C3 有「核对客户口径」步骤", any(n == "核对客户口径" for n, _ in v3.steps), True)
check("C3 不声称已登记", "已为您登记" in text3, False)
check("C3 引用运单尾号 8873", "8873" in text3, True)

print("\n=== 三、试一句：选单与不选单的对照（F18 验收）===")
v = run("买错色号了想换一个")
check("不选单 → need_info", v.state, "need_info")
check("不选单 → 缺订单号", [m.field for m in v.missing], ["订单号"])
v = run("买错色号了想换一个", "SO20261106012")
check("选 C8 单 → ready", v.state, "ready")
check("选 C8 单 → 售后规则-005", v.evidence[0]["kbId"], "售后规则-005")
v = run("显示签收但我没收到", "SO20261106012")
check("选单 → blocked", v.state, "blocked")
check("选单 → 物流规则-008", v.evidence[0]["kbId"], "物流规则-008")

print("\n=== 四、订单清单（list，主键 orderId）===")
check("订单数", len(ORDERS), 7)
for o in ORDERS:
    print(f"  {o['orderId']} conv={o.get('convId')} missing={o.get('missingFields')}")

print("\n=== 五、C9 对照：无关缺字段不拦本轮（F05 验收）===")
c9 = next(o for o in ORDERS if o.get("convId") == "C9")["orderId"]
v = run("这个开封后能用多久", c9)
check("问开封期限 → need_info", v.state, "need_info")
v = run("退款到账了吗", c9)
check("问退款 → ready（对照组）", v.state, "ready")

print("\n=== 六、冲突示例（F08 验收）===")
v = run("拆封用过的美妆还能退吗")
check("conflict", v.state, "conflict")
check("can_send=True", v.can_send, True)
check("requires_ack=True", v.requires_ack, True)
check("并列两条依据", len(v.evidence), 2)
check("冲突依据不含 high 风险", any(e.get("risk") == "high" for e in v.evidence), False)

print("\n=== 七、风险词前置（F02 验收）===")
v = run("我要退一赔三，不然就去 12315 投诉。你们到底什么时候给答复")
check("上一句提过风险词 → blocked", v.state, "blocked")
check("blocked 原因恒为 high_risk", v.handoff_reason, "high_risk")
kw = engine.HANDOFF_KEYWORDS
hi = {"12315", "投诉", "曝光", "律师", "消协", "过敏", "烂脸", "起痘", "刺痛", "赔偿", "赔钱"}
check("词表 16 个", len(kw), 16)
check("high 11 个", len([w for w in kw if w in hi]), 11)
check("medium 5 个", len([w for w in kw if w not in hi]), 5)
check("退一赔三不在词表", "退一赔三" in kw, False)

print("\n=== 八、标签与归属（唯一来源 policy）===")
print(" ", json.dumps(policy.STATE_LABEL, ensure_ascii=False))
print(" ", json.dumps(policy.OWNER_LABEL, ensure_ascii=False))
check("ready 标签", policy.STATE_LABEL["ready"], "可发建议")
check("conflict 归属", policy.OWNER_LABEL["conflict"], "客服（判断不了升级组长）")
check("error 标签", policy.STATE_LABEL["error"], "查询失败")
check("error 归属", policy.OWNER_LABEL["error"], "客服（保留人工入口）")

print("\n=== 九、知识库规模与口径 ===")
check("规则总数", len(kb), 31)
cats = {}
for r in kb:
    cats[r["category"]] = cats.get(r["category"], 0) + 1
print("  分类:", cats)
check("售后咨询 14", cats.get("售后咨询"), 14)
check("商品咨询 9", cats.get("商品咨询"), 9)
check("物流咨询 8", cats.get("物流咨询"), 8)
cg = [(r["id"], r["risk_level"], r["conflict_group"]) for r in kb if r.get("conflict_group")]
print("  互斥组:", cg)
check("只有 2 条带 conflict_group", len(cg), 2)
r3 = next(r for r in kb if r["id"] == "物流规则-003")
check("物流规则-003 已对齐 48 小时", "48 小时" in r3["conditions"], True)
check("物流规则-003 不再写 3 天", "3 天" in r3["conditions"], False)

print("\n=== 十、A 场景代价（含假设声明）===")
a = next(o for o in ORDERS if o.get("convId") == "C8")
reasons = {r.code: r for r in policy.return_reasons(a)}
for code, r in reasons.items():
    print(f"  {code} allowed={r.allowed} cost={r.cost_text} 品退率={r.counts_return_rate} 评分={r.affects_score}")
check("ASSUMED_SHIPPING = 12（假设值）", policy.ASSUMED_SHIPPING, 12)
check("七天无理由 allowed", reasons["seven_day"].allowed, True)
check("换货 allowed", reasons["exchange"].allowed, True)
check("品质退货被禁", reasons["quality_return"].allowed, False)
check("只有品质退货计品退率",
      [c for c, r in reasons.items() if r.counts_return_rate], ["quality_return"])
check("只有品质退货影响评分",
      [c for c, r in reasons.items() if r.affects_score], ["quality_return"])

print("\n=== 十一、评测 CSV 复算（engine.py 历史基线，不是 policy.py）===")
# 口径必须与 app.py 完全一致：utf-8-sig 读（文件带 BOM），布尔比较走 truthy()
# 见 app.py:177 / 181-182 / 1291-1296。自己另写一套比较就会算出不同的百分比。
def truthy(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "evaluation_results.csv"),
                                encoding="utf-8-sig")))
check("CSV 行数 = 50×3", len(rows), 150)
for ver in ["V1", "V2", "V3"]:
    rs = [r for r in rows if r["version"] == ver]
    n = len(rs)
    intent = sum(r["pred_intent"] == r["expected_intent"] for r in rs)
    handoff = sum(truthy(r["pred_handoff"]) == truthy(r["expected_handoff"]) for r in rs)
    risk = sum(r["pred_risk"] == r["expected_risk"] for r in rs)
    keyhit = sum(truthy(r["key_hit"]) for r in rs)
    # 无依据回答：answer 非空 且 refs 为空 且 未判转人工（app.py:1296 同口径）
    unfounded = sum(1 for r in rs if r["answer"].strip() and not r["refs"].strip()
                    and not truthy(r["pred_handoff"]))
    print(f"  {ver}: n={n} 意图={intent}/{n} 转人工={handoff}/{n} 风险={risk}/{n} "
          f"要点={keyhit}/{n} 无依据={unfounded}/{n}")
    check(f"{ver} 用例数", n, 50)
    check(f"{ver} 意图 41/50=82%", intent, 41)
    check(f"{ver} 无依据回答", unfounded, 50 if ver == "V1" else 0)
    check(f"{ver} 转人工", handoff, {"V1": 20, "V2": 33, "V3": 44}[ver])
    check(f"{ver} 风险判定", risk, {"V1": 20, "V2": 33, "V3": 44}[ver])
    check(f"{ver} 要点命中", keyhit, {"V1": 15, "V2": 24, "V3": 25}[ver])

print("\n=== 十二、C5 客户原话与时间戳自洽 ===")
c5msgs = [m for m in md["messages"] if m.get("convId") == "C5" and m["from"] == "customer"]
print("  C5 客户原话:", c5msgs[0]["text"] if c5msgs else "(无)")
check("C5 已改为「我上周买的」", "我上周买的" in c5msgs[0]["text"], True)
check("C5 不再写「前天买的」", "前天买的" in c5msgs[0]["text"], False)

print("\n=== 断言汇总 ===")
print(f"  {PASS} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("  失败项:", FAIL)
    sys.exit(1)
print("  全部一致：文档里的数字与状态均可由当前代码复算得到。")