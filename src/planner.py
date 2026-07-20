"""
地图加载与 A* 路径规划。
地图格式见 config/map.json。
"""

import json
import heapq
import math
from pathlib import Path
from typing import Optional


DEFAULT_MAP_PATH = Path(__file__).resolve().parents[1] / "config" / "map.json"


def load_map(map_path: str | Path = DEFAULT_MAP_PATH) -> dict:
    """加载地图 JSON，返回 {nodes, edges}。"""
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_adjacency(edges: list[dict]) -> dict[str, dict[str, float]]:
    """将边列表转为邻接表 {node: {neighbor: cost}}。"""
    adj: dict[str, dict[str, float]] = {}
    for e in edges:
        u, v, cost = e["from"], e["to"], e["dist_cm"]
        adj.setdefault(u, {})[v] = cost
        adj.setdefault(v, {})[u] = cost
    return adj


def astar(
    nodes: dict[str, dict],
    adj: dict[str, dict[str, float]],
    start: str,
    goal: str,
) -> Optional[list[str]]:
    """A* 搜索，返回节点 ID 序列，若无路径返回 None。"""

    def _heuristic(a: str, b: str) -> float:
        ax, ay = nodes[a]["x"], nodes[a]["y"]
        bx, by = nodes[b]["x"], nodes[b]["y"]
        return math.hypot(ax - bx, ay - by)

    open_set = [(0.0, start)]
    came_from: dict[str, str] = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for neighbor, cost in adj.get(current, {}).items():
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heapq.heappush(
                    open_set, (tentative + _heuristic(neighbor, goal), neighbor)
                )

    return None


def get_waypoints(
    map_data: dict, start: str, goal: str
) -> Optional[list[dict]]:
    """返回从 start 到 goal 的路径点列表，每个点包含节点 ID 和坐标。"""
    adj = build_adjacency(map_data["edges"])
    path = astar(map_data["nodes"], adj, start, goal)
    if path is None:
        return None
    return [
        {"id": nid, "x": map_data["nodes"][nid]["x"],
         "y": map_data["nodes"][nid]["y"]}
        for nid in path
    ]
