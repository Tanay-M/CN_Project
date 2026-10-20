import numpy as np
from test_routing import network

from carbonroute.common import config
from carbonroute.network import Network
from carbonroute.routing import BudgetPlanner, Planner


def test_joint_selector_never_uses_more_predicted_carbon_than_reactive_when_unheld():
    cfg = config()
    cfg.update(delay_stretch=10, hold_bins=0, min_improvement=0)
    graph = network()
    pairs = [(0, 3), (2, 3)]
    carbon = np.array([1, 10, 1, 1, 1, 1, 1, 1.])
    forecast = np.zeros((3, 8, 8))
    forecast[:, 0, 3] = 6
    forecast[:, 2, 3] = 5
    reactive = Planner(graph, pairs, cfg, "reactive_carbon").decide(forecast, carbon, 0)
    selected = BudgetPlanner(graph, pairs, cfg, 0.0).decide(forecast, carbon, 0, forecast)

    def weighted_cost(routes):
        return 6 * sum(carbon[n] for n in routes[0][:-1]) + 5 * sum(
            carbon[n] for n in routes[1][:-1])

    assert weighted_cost(selected.routes) <= weighted_cost(reactive.routes) + 1e-9
    assert selected.projected_max_utilization <= reactive.projected_max_utilization + 1e-9
    assert selected.carbon_budget_met is True


def test_joint_assignment_reduces_peak_within_reactive_budget_on_six_flows():
    # Fixed eight-node graph and worked traffic example; no dataset download needed.
    coordinates = [(2.351, 48.8566), (8.6842, 50.1122), (4.3518, 50.8469),
                   (4.9407, 52.3236), (6.1399, 46.2038), (9.19, 45.4642),
                   (16.3729, 48.2091), (6.1296, 49.6112)]
    edges = [(4, 6), (1, 6), (0, 2), (2, 7), (2, 3), (0, 4),
             (4, 5), (0, 1), (1, 5), (1, 3), (0, 7)]
    graph = Network(coordinates, edges, [str(i) for i in range(len(edges))], 10, 1.3, .15)
    pairs = [(5, 3), (5, 0), (1, 2), (3, 0), (3, 5), (4, 1)]
    demands = [3.04, 3.86, 3.3, 4.31, 5.3, 6.72]
    carbon = np.array([480, 199, 219, 403, 404, 429, 526, 203.])
    forecast = np.zeros((3, 8, 8))
    for (a, b), value in zip(pairs, demands):
        forecast[:, a, b] = value
    cfg = config()
    cfg.update(hold_bins=2, min_improvement=0)
    reactive = Planner(graph, pairs, cfg, "reactive_carbon").decide(forecast, carbon, 0)
    planner = BudgetPlanner(graph, pairs, cfg)
    selected = planner.decide(forecast, carbon, 0, forecast)
    assert selected.joint_assignments_evaluated == 64
    assert selected.carbon_budget_met is True
    assert selected.projected_max_utilization < reactive.projected_max_utilization
    # On the next bin, the reference assignment is cheaper in the new carbon mix.
    # The carbon budget must still hold even though a two-bin route hold is active.
    changed_carbon = np.array([501, 392, 287, 327, 189, 318, 252, 178.])
    following = planner.decide(forecast, changed_carbon, 1, forecast)
    assert following.carbon_budget_met is True


def test_joint_selector_enforces_budget_even_when_current_route_has_lower_peak():
    cfg = config()
    cfg.update(delay_stretch=10, hold_bins=0, min_improvement=0)
    graph = network()
    pairs = [(0, 3), (2, 3)]
    forecast = np.zeros((3, 8, 8))
    forecast[:, 0, 3] = 4.99
    forecast[:, 2, 3] = 2.73
    carbon = np.array([61, 66, 52, 93, 92, 21, 61, 63.])
    selected = BudgetPlanner(graph, pairs, cfg, 0.0).decide(forecast, carbon, 0, forecast)
    assert selected.routes == ((0, 2, 3), (2, 3))
    assert selected.carbon_budget_met is True
