"""POC 测试报告生成器：跑一遍预设问答集，自动产出客户可验收的测试报告。

这是本项目里唯一一个"把售前动作产品化"的模块——POC 是解决方案专家的日常交付物，
过去是手工整理 Excel，这里把它变成一条命令。

用法：
    python -m app.benchmark              # 完整测试（需要 Ollama 与本地模型）
    python -m app.benchmark --no-llm     # 只测检索与拒答闸门（秒级，不需要模型）
    python -m app.benchmark --out 报告.md # 指定报告输出路径

产出：reports/POC测试报告.md
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import rag
from .config import INDEX_DIR, LLM_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = Path(__file__).resolve().parent / "qa_set.json"
REPORT_DIR = PROJECT_ROOT / "reports"


def _compact(s: str) -> str:
    """去掉空白与常见分隔符，避免"30 分钟"和"30分钟"被判成不同。"""
    return "".join(ch for ch in s if not ch.isspace())


def hit_ratio(text: str, keywords: list[str]) -> float:
    """答案命中率：特征词有多少比例出现在回答里（忽略空格差异）。"""
    if not keywords:
        return 1.0
    body = _compact(text)
    return sum(1 for k in keywords if _compact(k) in body) / len(keywords)


def run_case(case: dict, use_llm: bool) -> dict:
    started = time.time()
    if use_llm:
        result = rag.answer(case["question"], scenario=case.get("scenario"))
        answer_text = result["answer"]
        confidence = result["confidence"]
        citations = result.get("citations") or []
    else:
        # 无模型模式：只跑到闸门，用检索结果模拟回答文本
        hits = rag.retrieve(case["question"], scenario=case.get("scenario"))
        confidence = rag.grade_confidence(hits)
        missing = rag.article_ref_missing(case["question"], hits)
        if missing:
            confidence = "无法回答"
        answer_text = "".join(h["text"] for h in hits[:3])  # 用原文冒充回答，用于测检索命中
        citations = [
            {"source": h["source"], "version": h["version"], "page": h["page"]} for h in hits
        ]
    latency = (time.time() - started) * 1000

    versions = {c["version"] for c in citations}
    if case.get("must_refuse"):
        passed = confidence == "无法回答"
        score_note = "拒答正确" if passed else "未拒答（风险）"
        ratio = 1.0 if passed else 0.0
    else:
        ratio = hit_ratio(answer_text, case.get("keywords", []))
        passed = ratio >= 0.6 and confidence != "无法回答"
        score_note = "命中 %.0f%%" % (ratio * 100)
        if confidence == "无法回答":
            score_note = "误拒答（本应回答）"

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "confidence": confidence,
        "expected_refuse": bool(case.get("must_refuse")),
        "passed": passed,
        "score": round(ratio, 2),
        "note": score_note,
        "latency_ms": int(latency),
        "versions_hit": len(versions),
        "expect_versions": case.get("expect_versions", 1),
        "answer_preview": answer_text[:120].replace("\n", " "),
    }


def summarize(rows: list[dict]) -> dict:
    normal = [r for r in rows if not r["expected_refuse"]]
    refusal = [r for r in rows if r["expected_refuse"]]
    lat = [r["latency_ms"] for r in rows]
    summary = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r["passed"]),
        "answer_accuracy": round(sum(1 for r in normal if r["passed"]) / len(normal) * 100, 1) if normal else 0.0,
        "refusal_accuracy": round(sum(1 for r in refusal if r["passed"]) / len(refusal) * 100, 1) if refusal else 0.0,
        "avg_latency_ms": int(statistics.mean(lat)) if lat else 0,
        "p95_latency_ms": int(sorted(lat)[int(len(lat) * 0.95) - 1]) if len(lat) > 1 else (lat[0] if lat else 0),
    }
    version_rows = [r for r in rows if r["expect_versions"] > 1]
    if version_rows:
        ok = sum(1 for r in version_rows if r["versions_hit"] >= r["expect_versions"])
        summary["version_coverage"] = round(ok / len(version_rows) * 100, 1)
    return summary


def build_report(summary: dict, rows: list[dict], stats: dict, use_llm: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    env = "%s · %s · %s" % (platform.system(), platform.machine(), platform.processor() or "Apple Silicon")
    failures = [r for r in rows if not r["passed"]]

    lines = [
        "# 政企制度智能问答系统 · POC 测试报告",
        "",
        "| 项目 | 内容 |",
        "|------|------|",
        "| 测试时间 | %s |" % now,
        "| 测试环境 | %s |" % env,
        "| 生成模型 | %s（本地 Ollama，数据不出域） |" % (LLM_MODEL if use_llm else "未启用（仅测检索）"),
        "| 知识库规模 | %d 个切片 / %d 维向量 |" % (stats.get("chunks", 0), stats.get("dim", 0)),
        "| 语料范围 | %s |" % "、".join(stats.get("sources", [])),
        "| 测试用例 | %d 条（常规 %d / 版本对比 %d / 操作辅助 %d / 应拒答 %d） |"
        % (
            summary["total"],
            sum(1 for r in rows if r["type"] == "常规问答"),
            sum(1 for r in rows if r["type"] == "版本对比"),
            sum(1 for r in rows if r["type"] == "操作辅助"),
            sum(1 for r in rows if r["type"] == "应拒答"),
        ),
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 结果 | 说明 |",
        "|------|------|------|",
        "| 用例通过率 | %d / %d（%.0f%%） | 逐条人工可复核 |"
        % (summary["passed"], summary["total"], summary["passed"] / summary["total"] * 100),
        "| 答案命中率 | %.1f%% | 应答题中，回答覆盖原文特征词的比例 |" % summary["answer_accuracy"],
        "| **拒答正确率** | **%.1f%%** | 超纲问题正确拒答的比例，政企场景的核心指标 |" % summary["refusal_accuracy"],
        "| 平均响应时延 | %d ms | %s |" % (summary["avg_latency_ms"], "含模型推理" if use_llm else "不含模型推理"),
        "| P95 响应时延 | %d ms | |" % summary["p95_latency_ms"],
    ]
    if "version_coverage" in summary:
        lines.append(
            "| 版本覆盖正确率 | %.1f%% | 版本对比题中，引用同时覆盖两个版本的比例 |" % summary["version_coverage"]
        )

    lines += ["", "## 二、逐条测试明细", "", "| 编号 | 类型 | 问题 | 置信度 | 结果 | 时延 |", "|------|------|------|--------|------|------|"]
    for r in rows:
        mark = "通过" if r["passed"] else "**未通过**"
        q = r["question"] if len(r["question"]) <= 30 else r["question"][:30] + "…"
        lines.append(
            "| %s | %s | %s | %s | %s（%s） | %d ms |"
            % (r["id"], r["type"], q, r["confidence"], mark, r["note"], r["latency_ms"])
        )

    if failures:
        lines += ["", "## 三、未通过用例分析", ""]
        for r in failures:
            lines.append("**%s · %s**" % (r["id"], r["question"]))
            lines.append("")
            lines.append("- 期望：%s；实际：%s（%s）" % ("拒答" if r["expected_refuse"] else "回答并命中特征词", r["confidence"], r["note"]))
            lines.append("- 实际输出片段：%s" % r["answer_preview"])
            lines.append("")
    else:
        lines += ["", "## 三、未通过用例分析", "", "本轮全部用例通过。"]

    lines += [
        "",
        "## 四、数据安全声明",
        "",
        "- 全部推理在本机完成（Ollama 本地服务），**不调用任何云端 API**，制度文档与提问内容不出本机。",
        "- 向量检索为本地文件索引（numpy + 本地 embedding 模型），无外部依赖。",
        "- 语料为公开法律法规与自编演示数据，不含任何客户或企业内部资料。",
        "",
        "## 五、已知边界",
        "",
        "- 检索质量受切分粒度与 embedding 模型能力限制，口语化、跨条款组合类问题命中率会下降。",
        "- 拒答依赖「相似度阈值 + 条款号校验 + 模型自述」三道闸，灰区问题仍有少量误判。",
        "- 单机单模型，未做并发压测；生产环境需补推理引擎选型与硬件容量规划。",
        "- **时延受宿主机负载影响显著**：同一环境实测过 9.6s 与 20.9s 两轮（后者是模型常驻内存 + 系统 swap 7.8GB 时），",
        "  端侧方案的容量规划必须按最差负载做预算，不能只看空载数据。",
        "- 「目的类」口语化问题（\"为什么编制\"）与法条措辞语义距离远，闸一阈值区分度有限，",
        "  由灰档机制兜底（cos 0.42~0.52 展示最相关条款原文，不生成答案）。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="跑 POC 测试集并生成报告")
    parser.add_argument("--no-llm", action="store_true", help="只测检索与拒答闸门，不调用大模型")
    parser.add_argument("--out", default=None,
                        help="报告输出路径（默认：完整模式 reports/POC测试报告.md，"
                             "仅检索模式 reports/POC测试报告_检索层.md，两种模式互不覆盖）")
    args = parser.parse_args()

    use_llm = not args.no_llm
    cases = json.loads(QA_PATH.read_text(encoding="utf-8"))["cases"]
    stats = rag.knowledge_stats()

    print("[POC] 共 %d 条用例，模式：%s" % (len(cases), "完整（含模型推理）" if use_llm else "仅检索"))
    rows = []
    for case in cases:
        row = run_case(case, use_llm)
        rows.append(row)
        print("  %s %-8s %s ｜ %s ｜ %dms" % (row["id"], row["type"], "通过" if row["passed"] else "未过", row["note"], row["latency_ms"]))

    summary = summarize(rows)
    report = build_report(summary, rows, stats, use_llm)

    # 两种模式输出到不同文件，避免 --no-llm 快速自检覆盖掉含模型推理的正式报告
    out = Path(args.out) if args.out else REPORT_DIR / (
        "POC测试报告.md" if use_llm else "POC测试报告_检索层.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")

    print("\n[POC] 通过 %d/%d ｜ 答案命中 %.1f%% ｜ 拒答正确 %.1f%% ｜ 平均时延 %dms"
          % (summary["passed"], summary["total"], summary["answer_accuracy"],
             summary["refusal_accuracy"], summary["avg_latency_ms"]))
    print("[POC] 报告已生成：%s" % out)


if __name__ == "__main__":
    main()
