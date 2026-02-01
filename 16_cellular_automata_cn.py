# -*- coding: utf-8 -*-
"""
细胞自动机/网格智能体（Cellular Automata / Grid-Based Systems）架构的可运行示例

学习目标（看完你能做到什么）：
- 理解「大量简单单元 + 局部规则」涌现出路径规划等全局行为
- 掌握网格初始化、同步 tick、梯度传播与沿梯度下降遍历
- 能运行仓库拣货演示：从打包站扩散路径波，再从货架沿梯度走到打包站

核心概念速览：
- 网格智能体：每个格子是一个小智能体，仅根据邻居状态按规则更新自身状态
- 路径波传播：从目标格（打包站）将 pathfinding_value=0 向外扩散，形成到各格的最短步数梯度
- 沿梯度下降：从货架格沿「值递减」方向走到打包站，即拣货路径

运行前准备：
- 项目根目录创建 `.env`（可选，本脚本主要用 numpy + rich，无需 MODELSCOPE_API_KEY 即可跑网格演示）
- 若需结尾 LLM 总结：配置 `MODELSCOPE_API_KEY`

如何运行：
- 直接运行默认示例：`python 16_cellular_automata_cn.py`
- 指定拣货清单：`python 16_cellular_automata_cn.py --order A,B`
- 智能体群组演示（打印每 tick 各格子的更新过程）：`python 16_cellular_automata_cn.py --verbose`

运行效果示例（Rich 美化输出）：
    --- 初始网格 ---
    ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
    │·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
    │·│·│█│·│·│·│·│·│D│·│·│·│·│·│5│·│·│·│·│·│6│·│·│·│█│·│
    │·│·│█│·│·│·│A│·│4│·│·│3│·│·│2│·│·│C│·│·│6│·│·│7│·│·│
    │·│·│█│·│·│·│5│·│·│4│·│·│3│·│·│4│·│·│5│·│·│B│·│8│·│·│
    │·│·│█│·│·│6│·│·│5│·│·│4│·│·│P│·│·│6│·│·│7│·│·│8│·│·│
    │·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
    └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
    拣货 A：路径 (2,6)→(3,6)→...→(4,14)；拣货 B：路径 (3,21)→...→(4,14)
    可选 LLM 总结：仓库网格拣货系统已根据订单 ['A','B'] 完成拣货，商品 A 和 B 均汇合于点 (4,14)。

阅读建议：
- 先看「网格与单元规则」，再看「路径波传播」与「沿梯度遍历」，最后看主流程。
"""

import os
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# =========================
# 1) LLM 初始化
# =========================

console = Console()
DEBUG: bool = False


class ModelScopeChat:
    """
    ModelScope 的 OpenAI 兼容接口，与项目其他 py 一致。
    仅 invoke 纯文本，用于可选 LLM 总结。
    """
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None, temperature: float = 0.2, extra_body: Optional[dict] = None):
        self.base_url = base_url or os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
        self.api_key = api_key or os.environ.get("MODELSCOPE_API_KEY", "")
        self.model = model or os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
        self.temperature = temperature
        self.extra_body = extra_body or {"enable_thinking": True, "trust_request_chat_template": True}
        self.client = None
        if self.api_key:
            from openai import OpenAI
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def invoke(self, prompt: str) -> str:
        if not self.client:
            return ""
        extra = dict(self.extra_body) if self.extra_body else {}
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                stream=False,
                extra_body=extra,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if DEBUG:
                console.print(f"[dim]LLM 调用失败: {e}[/dim]")
            return ""


def init_llm() -> ModelScopeChat:
    """从 .env 读取 ModelScope 配置。"""
    base_url = os.environ.get("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
    api_key = os.environ.get("MODELSCOPE_API_KEY", "")
    model_id = os.environ.get("MODELSCOPE_MODEL_ID", "deepseek-ai/DeepSeek-V3.2")
    extra = {"enable_thinking": True, "trust_request_chat_template": True}
    return ModelScopeChat(base_url=base_url, api_key=api_key, model=model_id, temperature=0.2, extra_body=extra)


# =========================
# 2) 网格与单元定义
# =========================

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


class CellAgent:
    """
    网格中的单格，相当于一个「小智能体」。
    每个格子只知道：自己的类型、自己的 pathfinding_value、邻居的 pathfinding_value。
    通过简单的局部规则，整体会涌现出「到打包站的最短路径梯度」。
    """
    def __init__(self, cell_type: str, item: Optional[str] = None):
        self.type = cell_type  # 空位、障碍、货架、打包站
        self.item = item
        self.pathfinding_value = float("inf")  # 初始无穷大，表示「还不知道离打包站多远」

    def update_value(self, neighbors: List["CellAgent"]) -> float:
        """
        局部规则：我离打包站多远？

        直觉：打包站 P 离自己是 0 步；P 旁边的格子离 P 是 1 步；再外层是 2 步……
        所以：如果我的某个邻居离 P 是 k 步，那我离 P 最多是 k+1 步（从邻居走一步过来）。
        取「邻居里最小的 k」再 +1，就是我最新的可能步数。
        障碍格永远不通，返回无穷大。

        公式：新值 = min(当前值, 1 + min(邻居们的值))
        """
        if self.type == "障碍":
            return float("inf")
        min_n = float("inf")
        for n in neighbors:
            if n.pathfinding_value < min_n:
                min_n = n.pathfinding_value
        return min(self.pathfinding_value, min_n + 1)


class WarehouseGrid:
    """仓库网格：从布局字符串构建，支持 get_neighbors、tick、visualize。
    其中get_neighbors 用于获取邻居格子，tick 用于更新所有格子的 pathfinding_value，visualize 用于可视化网格。
    """
    def __init__(self, layout: List[str]):
        if not np:
            raise RuntimeError("需要 numpy，请执行: pip install numpy")
        self.height = len(layout)
        self.width = len(layout[0])
        self.grid = self._create_grid(layout)
        self.item_locations = self._get_item_locations()

    def _create_grid(self, layout: List[str]):
        grid = np.empty((self.height, self.width), dtype=object)
        # 创建网格，根据布局字符串创建格子，空位、障碍、打包站、货架分别用不同的 CellAgent 实例表示。
        for r in range(self.height):
            row = layout[r] if r < len(layout) else ""
            for c in range(self.width):
                ch = row[c] if c < len(row) else " "
                if ch == " ":
                    grid[r, c] = CellAgent("空位")
                elif ch == "#":
                    grid[r, c] = CellAgent("障碍")
                elif ch == "P":
                    grid[r, c] = CellAgent("打包站")
                else:
                    grid[r, c] = CellAgent("货架", item=ch)
        return grid

    def _get_item_locations(self) -> Dict[str, Tuple[int, int]]:
        """获取所有货架和打包站的位置。"""
        loc = {}
        for r in range(self.height):
            for c in range(self.width):
                cell = self.grid[r, c]
                if cell is None:
                    continue
                if cell.type == "货架":
                    loc[cell.item] = (r, c)
                if cell.type == "打包站":
                    loc["P"] = (r, c)
        return loc

    def get_neighbors(self, r: int, c: int) -> List[CellAgent]:
        """获取邻居格子。"""
        out = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                out.append(self.grid[nr, nc])
        return out

    def tick(self, verbose: bool = False) -> bool:
        """更新所有格子的 pathfinding_value。"""
        new_val = np.empty((self.height, self.width))
        changed_info = []  # (r, c, old_v, new_v, min_n) 用于 verbose，在应用前捕获邻居旧值
        for r in range(self.height):
            for c in range(self.width):
                neighbors = self.get_neighbors(r, c)
                new_val[r, c] = self.grid[r, c].update_value(neighbors)
                old_v = self.grid[r, c].pathfinding_value
                if old_v != new_val[r, c] and verbose:
                    min_n = min((n.pathfinding_value for n in neighbors), default=float("inf"))
                    changed_info.append((r, c, old_v, new_val[r, c], min_n))
        changed = False
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c].pathfinding_value != new_val[r, c]:
                    self.grid[r, c].pathfinding_value = new_val[r, c]
                    changed = True
        if verbose and changed_info:
            for r, c, old_v, new_v, min_n in changed_info[:15]:
                old_s = "inf" if old_v == float("inf") else int(old_v)
                new_s = "inf" if new_v == float("inf") else int(new_v)
                min_s = "inf" if min_n == float("inf") else int(min_n)
                cell = self.grid[r, c]
                label = cell.item if cell.type == "货架" else ("P" if cell.type == "打包站" else "·")
                console.print(f"    [dim]智能体({r},{c})[/dim] {label} 邻居 min={min_s} → 更新 {old_s}→{new_s}")
            if len(changed_info) > 15:
                console.print(f"    [dim]... 本 tick 共 {len(changed_info)} 个智能体更新[/dim]")
        return changed

    def visualize(self, show_values: bool = False, title: str = "仓库网格"):
        """可视化网格。"""
        table = Table(title=title, show_header=False, show_edge=True, padding=0)
        for _ in range(self.width):
            table.add_column(justify="center")
        for r in range(self.height):
            row = []
            for c in range(self.width):
                cell = self.grid[r, c]
                if cell.type == "空位":
                    row.append("[grey70]·[/grey70]")
                elif cell.type == "障碍":
                    row.append("[red]█[/red]")
                elif cell.type == "打包站":
                    row.append("[bold green]P[/bold green]")
                elif cell.type == "货架":
                    row.append(f"[bold blue]{cell.item}[/bold blue]")
                else:
                    row.append("?")
                if show_values and cell.pathfinding_value != float("inf"):
                    row[-1] = f"{int(cell.pathfinding_value)}"
            table.add_row(*row)
        console.print(table)


# =========================
# 3) 路径波传播与沿梯度遍历
# =========================

def propagate_path_wave(grid: WarehouseGrid, target_pos: Tuple[int, int], verbose: bool = False) -> None:
    """
    路径波传播：从目标格（打包站）将 pathfinding_value=0 向外扩散，形成到各格的最短步数梯度。
    具体做法：
    1. 初始化所有格子的 pathfinding_value 为无穷大。
    2. 将目标格的 pathfinding_value 设置为 0。
    3. 重复执行 tick，直到所有格子的 pathfinding_value 不再变化（梯度稳定）。

    其中 target_pos 为目标格坐标，verbose 为 True 时打印每 tick 各格子的更新过程。
    """
    r0, c0 = target_pos
    for r in range(grid.height):
        for c in range(grid.width):
            grid.grid[r, c].pathfinding_value = float("inf")
    grid.grid[r0, c0].pathfinding_value = 0
    tick_count = 0
    while True:
        tick_count += 1
        if verbose:
            console.print(f"  [bold cyan]智能体群组 tick #{tick_count}[/bold cyan]")
        if not grid.tick(verbose=verbose):
            break


def trace_path(grid: WarehouseGrid, start_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    沿梯度遍历：从起点格沿「值递减」方向走到目标格，即拣货路径。
    具体做法：
    1. 初始化路径为空。
    2. 从起点格开始，沿「值递减」方向找到下一个格子。
    3. 将下一个格子添加到路径中。
    4. 重复步骤 2 和 3，直到到达目标格（pathfinding_value=0）。
    5. 返回路径。
    
    其中 start_pos 为起点格坐标，grid 为网格。
    """
    path = [start_pos]
    r, c = start_pos
    while grid.grid[r, c].pathfinding_value > 0:
        best = None
        best_val = grid.grid[r, c].pathfinding_value
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= nr < grid.height and 0 <= nc < grid.width:
                cell = grid.grid[nr, nc]
                if cell.pathfinding_value < best_val:
                    best_val = cell.pathfinding_value
                    best = (nr, nc)
        if best is None:
            break
        path.append(best)
        r, c = best
    return path


# =========================
# 4) 默认布局与拣货流程
# =========================

DEFAULT_LAYOUT = [
    "  #   #   #   #   #   #  #",
    "  #     D     5     6   #",
    "  #   A 4  3  2  C  6  7",
    "  #   5  4  3  4  5  B 8",
    "  #  6  5  4  P  6  7  8",
    "  #   #   #   #   #   #  #",
]# 布局字符说明（字符串 → 可视化符号）：
#   空格 " "  → · 空位（可通行）
#   "#"      → █ 障碍（墙，不可通过）
#   "P"      → P 打包站（目标点，拣货后送达）
#   其他字符  → 货架（A/B/C/D/2/3/4/5/6/7/8 为商品编号）
# 布局可视化示例：
# ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
# │·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
# │·│·│█│·│·│·│·│·│D│·│·│·│·│·│5│·│·│·│·│·│6│·│·│·│█│·│
# │·│·│█│·│·│·│A│·│4│·│·│3│·│·│2│·│·│C│·│·│6│·│·│7│·│·│
# │·│·│█│·│·│·│5│·│·│4│·│·│3│·│·│4│·│·│5│·│·│B│·│8│·│·│
# │·│·│█│·│·│6│·│·│5│·│·│4│·│·│P│·│·│6│·│·│7│·│·│8│·│·│
# │·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│·│█│·│·│█│
# └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

def fulfill_order(grid: WarehouseGrid, order: List[str], verbose: bool = False) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """
    拣货流程：
    1. 找到打包站 P。
    2. 遍历订单中的每个商品，找到其位置。
    3. 从打包站扩散路径波到商品位置。
    4. 从商品位置沿梯度遍历到打包站。
    5. 返回路径。
    
    其中grid 为网格，order 为订单，verbose 用于打印每步的更新过程。
    """
    packing = grid.item_locations.get("P")
    if not packing:
        console.print("[red]未找到打包站 P[/red]")
        return []
    results = []
    for item in order:
        if item not in grid.item_locations:
            console.print(f"[yellow]货架无商品 {item}，跳过[/yellow]")
            continue
        title = f"拣货：{item}" + (" [智能体群组演示]" if verbose else "")
        console.print(Panel("从打包站扩散路径波…", title=title, border_style="cyan"))
        if verbose:
            console.print("  [dim]各格子根据邻居同步更新 pathfinding_value，波从 P 向外扩散[/dim]")
        propagate_path_wave(grid, packing, verbose=verbose)
        start = grid.item_locations[item]
        path = trace_path(grid, start)
        console.print(Panel(f"路径：{' -> '.join(str(p) for p in path)}", title=f"[OK] {item} 已拣", border_style="green"))
        results.append((item, path))
    return results


# =========================
# 5) CLI 与入口
# =========================

def parse_args():
    p = argparse.ArgumentParser(description="细胞自动机：仓库拣货路径演示")
    p.add_argument("--order", type=str, default="A,B", help="拣货清单，逗号分隔，如 A,B")
    p.add_argument("--show-values", action="store_true", help="可视化时显示路径值")
    p.add_argument("--verbose", action="store_true", help="智能体群组演示：打印每 tick 各格子的「邻居→更新」过程")
    p.add_argument("--debug", action="store_true", help="调试输出")
    return p.parse_args()


def main():
    global DEBUG
    load_dotenv()
    args = parse_args()
    DEBUG = getattr(args, "debug", False)
    if not np:
        console.print("[red]请安装 numpy: pip install numpy[/red]")
        return
    # 创建网格
    grid = WarehouseGrid(DEFAULT_LAYOUT)
    console.print("--- 初始网格 ---")
    grid.visualize(show_values=args.show_values)  # show_values 为 True 时显示路径值
    # 解析订单
    order = [x.strip() for x in args.order.split(",") if x.strip()]
    if not order:
        order = ["A", "B"]
    console.print(f"\n--- 拣货清单：{order} ---")
    # 完成拣货
    results = fulfill_order(grid, order, verbose=getattr(args, "verbose", False))
    console.print(Panel(
        f"订单 {order} 已通过网格路径波完成拣货。",
        title="[完成] 拣货完成",
        border_style="green",
    ))
    # 可选 LLM 总结
    if results:
        llm = init_llm()
        if llm.client:
            summary_prompt = (
                f"用一两句话总结：仓库网格拣货系统已根据订单 {order} 完成拣货，"
                f"各商品路径为：{results}。不要输出代码。"
            )
            summary = llm.invoke(summary_prompt)
            if summary:
                console.print("\n--- LLM 总结 ---")
                console.print(summary)
        elif not os.environ.get("MODELSCOPE_API_KEY"):
            console.print("[dim]未配置 MODELSCOPE_API_KEY，跳过 LLM 总结[/dim]")


if __name__ == "__main__":
    main()
