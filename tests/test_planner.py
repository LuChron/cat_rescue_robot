from src.planner import astar, build_adjacency


def test_astar_does_not_mix_display_coordinates_with_edge_costs():
    nodes = {
        "start": {"x": 0, "y": 0},
        "detour": {"x": 10000, "y": 0},
        "goal": {"x": 1, "y": 0},
    }
    edges = [
        {"from": "start", "to": "goal", "dist_cm": 100},
        {"from": "start", "to": "detour", "dist_cm": 1},
        {"from": "detour", "to": "goal", "dist_cm": 1},
    ]

    assert astar(nodes, build_adjacency(edges), "start", "goal") == [
        "start",
        "detour",
        "goal",
    ]
