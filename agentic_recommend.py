# -*- coding: utf-8 -*-
"""
智能体架构推荐器（LLM 驱动）

根据用户需求描述，由 LLM 智能推荐最合适的智能体架构，并给出可直接落地的运行命令。

使用方式：
- 推荐架构：python agentic_recommend.py recommend "我需要医疗分诊，能识别急症转人工"
- 运行架构：python agentic_recommend.py run 17 --request "布洛芬和赖诺普利能一起吃吗？"
- 列出全部：python agentic_recommend.py list

推荐功能需配置 .env 中的 MODELSCOPE_API_KEY。
"""

import os
import sys
import json
import re
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# 架构注册表：(id, 名称, 文件, 典型场景)
ARCH_REGISTRY: List[Tuple[int, str, str, str]] = [
    (1, "反思（Reflection）", "01_reflection.py", "代码生成、复杂摘要、生成→评审→改写"),
    (2, "工具使用（Tool Use）", "02_tool_use.py", "API/函数调用、规划→执行→汇总"),
    (3, "ReAct", "03_react.py", "多步推理、边想边做、工具调用循环、Web 导航"),
    (4, "规划（Planning）", "04_planning.py", "报告生成、项目管理、先计划再执行"),
    (5, "多智能体（Multi-Agent）", "05_multi_agent.py", "专家团队、流水线协作、固定顺序"),
    (6, "规划-执行-验证（PEV）", "06_planner_executor_verifier.py", "自纠错、高风险自动化、金融"),
    (7, "黑板系统（Blackboard）", "07_blackboard.py", "动态协作、机会主义、复杂诊断"),
    (8, "情景+语义记忆", "08_episodic_with_semantic_cn.py", "长期个性化、跨会话、向量+图存储"),
    (9, "思维树（Tree of Thoughts）", "09_tree_of_thoughts_cn.py", "多路径探索、剪枝、逻辑谜题"),
    (10, "思维模型循环（Mental Loop）", "10_mental_loop_cn.py", "模拟→评估→执行、量化交易、机器人"),
    (11, "元控制器（Meta-Controller）", "11_meta_controller_cn.py", "按请求类型路由到专家、多服务入口"),
    (12, "图/世界模型记忆（Graph）", "12_graph_cn.py", "知识图谱、Cypher、多跳问答"),
    (13, "并行探索+集成（Ensemble）", "13_ensemble_cn.py", "多路分析、投资委员会、高 stakes 决策"),
    (14, "试跑外壳（Dry-Run）", "14_dry_run_cn.py", "发帖/发邮件前试跑、人工 approve 后再执行"),
    (15, "自改进循环（Self-Refine）", "15_RLHF_cn.py", "营销邮件、文档打磨、生成→评审→修订 until 通过"),
    (16, "细胞自动机/网格智能体", "16_cellular_automata_cn.py", "仓库拣货、路径规划、网格涌现"),
    (17, "反思式元认知（Reflexive Metacognitive）", "17_reflexive_metacognitive_cn.py", "医疗分诊、自知边界、直接回答/用工具/升级人工"),
]

# 各架构的正确 CLI 参数名（用于生成可运行的 run_command）
CLI_ARG_MAP = {
    1: ("--request", "用户请求"),
    2: ("--request", "用户请求"),
    3: ("--question", "用户问题"),  # 03 用 --question，不是 --request
    4: ("--request", "用户请求"),
    5: ("--company", "NVIDIA (NVDA)"),  # 05 用 --company
    6: ("--request", "用户请求"),
    7: ("--query", "用户查询"),
    8: ("--request", "用户请求"),
    9: ("--request", "用户请求"),
    10: ("--request", "用户请求"),
    11: ("--request", "用户请求"),
    12: ("--question", "用户问题"),  # 12 用 --question
    13: ("--request", "用户请求"),
    14: ("--request", "用户请求"),
    15: ("--request", "用户请求"),
    16: ("--order", "A,B"),  # 16 用 --order，可加 --verbose
    17: ("--request", "用户问题"),
}


def _build_run_command(arch_id: int, user_desc: str) -> str:
    """根据 arch_id 和用户描述，生成正确的可运行命令。"""
    if arch_id == 16:
        return f"python agentic_recommend.py run 16 --order A,B --verbose"
    if arch_id == 5:
        return f"python agentic_recommend.py run 5 --company \"NVIDIA (NVDA)\""
    arg_name, _ = CLI_ARG_MAP.get(arch_id, ("--request", "用户请求"))
    # 精简用户描述作为请求内容（过长会影响命令行）
    hint = user_desc[:50] + "…" if len(user_desc) > 50 else user_desc
    hint = hint.replace('"', "'")
    return f"python agentic_recommend.py run {arch_id} {arg_name} \"{hint}\""


class ArchRecommendation(BaseModel):
    """LLM 推荐结果。run_command 由系统根据 arch_id 生成，保证参数正确。"""
    arch_id: int = Field(ge=1, le=17, description="推荐的架构编号 1-17")
    reason: str = Field(description="推荐理由，贴心说明为何适合用户需求")
    run_command: str = ""  # 由 _build_run_command 填充，不依赖 LLM


def _get_arch_summary() -> str:
    """生成供 LLM 参考的架构摘要。"""
    lines = []
    for aid, name, fname, scene in ARCH_REGISTRY:
        lines.append(f"  #{aid:02d} {name}：{scene}（文件 {fname}）")
    return "\n".join(lines)


def _init_llm():
    """初始化 LLM 客户端。"""
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        model = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        client = OpenAI(base_url=base_url, api_key=api_key)
        return {"client": client, "model": model}
    except Exception:
        return None


def recommend_llm(description: str) -> Optional[ArchRecommendation]:
    """
    基于 LLM 的智能推荐。
    返回 ArchRecommendation 或 None（API 不可用时）。
    """
    llm_cfg = _init_llm()
    if not llm_cfg:
        return None
    client = llm_cfg["client"]
    model = llm_cfg["model"]
    arch_summary = _get_arch_summary()

    system = f"""你是智能体架构选型顾问。用户描述需求后，你从以下 17 种架构中选出最合适的一种，并给出贴心理由与可直接运行的命令。

可选架构（编号 1-17）：
{arch_summary}

输出要求：严格输出一个 JSON 对象，包含：
- arch_id：推荐的架构编号（1-17 的整数）
- reason：推荐理由，贴心、具体。必须包含两点：(1) 为何该架构最适合；(2) 明确说明「本推荐为可运行示例/模板，需接入自己的工具、数据、业务逻辑后落地到产品（二创）」。
不要输出 run_command，由系统自动生成。
不要输出任何解释或代码块标记，只输出 JSON。"""

    user = f"用户需求：{description}\n\n请推荐最合适的架构，并给出 arch_id 和 reason。"

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            stream=False,
            extra_body={"enable_thinking": True, "trust_request_chat_template": True, "response_format": {"type": "json_object"}},
        )
        raw = (r.choices[0].message.content or "").strip()
        # 提取 JSON
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0)) if m else {}
        if not isinstance(data, dict):
            data = {}
        # 清洗 arch_id
        aid = data.get("arch_id")
        if isinstance(aid, str) and aid.isdigit():
            data["arch_id"] = int(aid)
        elif isinstance(aid, float):
            data["arch_id"] = int(aid)
        rec = ArchRecommendation.model_validate({k: v for k, v in data.items() if k in ("arch_id", "reason", "run_command")})
        # 由系统生成正确的 run_command（避免 LLM 生成错误的参数名，如 03 用 --question 非 --request）
        rec.run_command = _build_run_command(rec.arch_id, description)
        return rec
    except (ValidationError, json.JSONDecodeError, KeyError) as e:
        print(f"[警告] LLM 返回解析失败: {e}，将使用兜底推荐。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[错误] LLM 调用失败: {e}", file=sys.stderr)
        return None


def run_arch(arch_id: int, extra_args: Optional[List[str]] = None) -> int:
    """运行指定架构的 py 文件。"""
    for aid, _, fname, _ in ARCH_REGISTRY:
        if aid == arch_id:
            py_path = ROOT / fname
            if not py_path.exists():
                print(f"[错误] 文件不存在: {py_path}", file=sys.stderr)
                return 1
            cmd = [sys.executable, str(py_path)] + (extra_args or [])
            return subprocess.call(cmd, cwd=str(ROOT))
    print(f"[错误] 未找到架构 #{arch_id}", file=sys.stderr)
    return 1


def main():
    p = argparse.ArgumentParser(
        description="智能体架构推荐器（LLM 驱动）：根据需求智能推荐并给出可落地运行命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python agentic_recommend.py recommend "我需要医疗分诊，能识别急症转人工"
  python agentic_recommend.py recommend "给我们的 AI 产品写营销邮件，要迭代打磨"
  python agentic_recommend.py run 17 --request "布洛芬和赖诺普利能一起吃吗？"
  python agentic_recommend.py list
        """,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("recommend", help="根据需求描述，由 LLM 智能推荐架构")
    rec.add_argument("description", type=str, nargs="?", default="", help="需求描述")

    run_p = sub.add_parser("run", help="运行指定架构")
    run_p.add_argument("arch_id", type=int, help="架构编号 01-17")
    run_p.add_argument("extra", nargs=argparse.REMAINDER, help="传递给架构脚本的参数")

    sub.add_parser("list", help="列出全部 17 种架构")

    args = p.parse_args()

    if args.cmd == "recommend":
        desc = args.description or input("请输入您的需求描述: ").strip() or "智能体"
        if not os.environ.get("MODELSCOPE_API_KEY"):
            print("\n[错误] 推荐功能需要配置 MODELSCOPE_API_KEY（.env 中）", file=sys.stderr)
            print("  python agentic_recommend.py list  可查看全部架构后手动选择\n", file=sys.stderr)
            return 1
        result = recommend_llm(desc)
        if not result:
            print("\n[错误] 推荐失败，请检查 API 或重试。可用 list 查看全部架构。\n", file=sys.stderr)
            return 1
        aid = result.arch_id
        name = next((n for i, n, _, _ in ARCH_REGISTRY if i == aid), f"#{aid}")
        print(f"\n根据「{desc}」的智能推荐：\n")
        print(f"  [推荐] #{aid:02d} {name}\n")
        print(f"  [理由] {result.reason}\n")
        print(f"  [运行] {result.run_command}")
        print(f"\n  [说明] 上列为可运行示例，需接入自己的工具、数据、业务逻辑后落地到产品（二创）。\n")
        return 0

    if args.cmd == "run":
        extra = getattr(args, "extra", None) or []
        extra = [x for x in extra if x != "--"]
        return run_arch(args.arch_id, extra)

    if args.cmd == "list":
        print("\n全部 17 种智能体架构：\n")
        for aid, name, fname, scene in ARCH_REGISTRY:
            print(f"  #{aid:02d} {name}")
            print(f"       {scene}  → {fname}\n")
        print("推荐: python agentic_recommend.py recommend \"您的需求\"")
        print("运行: python agentic_recommend.py run <编号> [参数]\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
