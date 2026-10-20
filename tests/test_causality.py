import numpy as np
from test_routing import network

from carbonroute.common import config
from carbonroute.data import Prepared
from carbonroute.prediction import Persistence
from carbonroute.simulation import replay


def test_changing_unobserved_demand_cannot_change_committed_route():
    cfg = config()
    cfg.update(lookback=2, delay_stretch=10)
    times = np.datetime64("2005-01-01T00:00") + np.arange(6)*np.timedelta64(15, "m")
    rates = np.zeros((6, 8, 8))
    rates[:, 0, 3] = 3
    mask = np.zeros((8, 8))
    mask[0, 3] = 1
    data = Prepared(times, rates, mask, np.ones((8, 8)), network(), [(0, 3)], {})
    carbon = np.tile([1, 500, 1, 100, 1, 1, 1, 1.], (4, 1))
    predictor = Persistence(3, mask)
    original = replay(data, cfg, rates, carbon, 2, 6, "predictive_carbon", predictor)
    changed = rates.copy()
    changed[2:, 0, 3] = 100
    counterfactual = replay(data, cfg, changed, carbon, 2, 6, "predictive_carbon", predictor)
    assert original.records[0]["routes"] == counterfactual.records[0]["routes"]
    assert original.records[0]["projected_max_utilization"] == counterfactual.records[0]["projected_max_utilization"]
    assert original.records[0]["served_mbps"] != counterfactual.records[0]["served_mbps"]
    assert original.records[0]["observed_through"] == "2005-01-01T00:15"


def test_joint_policy_commits_before_unobserved_demand_is_revealed():
    cfg = config()
    cfg.update(lookback=2, delay_stretch=10)
    times = np.datetime64("2005-01-01T00:00") + np.arange(6)*np.timedelta64(15, "m")
    rates = np.zeros((6, 8, 8))
    rates[:, 0, 3] = 3
    mask = np.zeros((8, 8))
    mask[0, 3] = 1
    data = Prepared(times, rates, mask, np.ones((8, 8)), network(), [(0, 3)], {})
    carbon = np.tile([1, 500, 1, 100, 1, 1, 1, 1.], (4, 1))
    predictor = Persistence(3, mask)
    original = replay(data, cfg, rates, carbon, 2, 6, "joint_gnn", predictor)
    changed = rates.copy()
    changed[2:, 0, 3] = 100
    counterfactual = replay(data, cfg, changed, carbon, 2, 6, "joint_gnn", predictor)
    assert original.records[0]["routes"] == counterfactual.records[0]["routes"]
    assert original.records[0]["reactive_carbon_budget"] == counterfactual.records[0]["reactive_carbon_budget"]
    assert original.records[0]["served_mbps"] != counterfactual.records[0]["served_mbps"]
