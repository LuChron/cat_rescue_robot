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
    if start not in nodes or goal not in nodes:
        return None

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


def get_cat_zones(map_data: dict) -> list[str]:
    """返回所有 type=cat_zone 的节点 ID 列表。"""
    return [
        nid for nid, node in map_data["nodes"].items()
        if node.get("type") == "cat_zone"
    ]


def plan_exploration_route(
    map_data: dict, start: str
) -> Optional[list[dict]]:
    """
    从 start 出发，贪心遍历所有猫区后返回 start。
    每次去最近的还没去过的猫区，最后回到 start。
    返回完整路径点列表（经过所有 cat_zone + 回到 start）。
    """
    adj = build_adjacency(map_data["edges"])
    cat_zones = get_cat_zones(map_data)
    if not cat_zones:
        return None

    current = start
    unvisited = set(cat_zones)
    full_route: list[dict] = [
        {"id": start, "x": map_data["nodes"][start]["x"],
         "y": map_data["nodes"][start]["y"]}
    ]

    while unvisited:
        # 找最近的未访问猫区
        best_zone = None
        best_dist = float("inf")
        for z in unvisited:
            zx, zy = map_data["nodes"][z]["x"], map_data["nodes"][z]["y"]
            cx, cy = map_data["nodes"][current]["x"], map_data["nodes"][current]["y"]
            d = math.hypot(zx - cx, zy - cy)
            if d < best_dist:
                best_dist = d
                best_zone = z

        # A* 去那个猫区
        path = astar(map_data["nodes"], adj, current, best_zone)
        if path is None:
            # 到不了就跳过，尝试下一个
            unvisited.discard(best_zone)
            continue

        # 添加路径点（跳过第一个，因为 current 已经在 route 里了）
        for nid in path[1:]:
            node = map_data["nodes"][nid]
            full_route.append({"id": nid, "x": node["x"], "y": node["y"]})

        current = best_zone
        unvisited.discard(best_zone)

    # 最后回到 start
    if current != start:
        path_back = astar(map_data["nodes"], adj, current, start)
        if path_back:
            for nid in path_back[1:]:
                node = map_data["nodes"][nid]
                full_route.append({"id": nid, "x": node["x"], "y": node["y"]})

    return full_route
