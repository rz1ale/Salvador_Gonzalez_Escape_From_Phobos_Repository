"""Competition accounting. These constants are not policy tuning weights.

Source: supplied official Scoring.py and JetSelectionActuator.py, version
2026/10/30. A fitted allocation period is a MODEL parameter, not native timing.
"""
from __future__ import annotations
import numpy as np

ISP_S = 80.0
G0_M_S2 = 9.8
NOMINAL_TIME_S = 120.0
NOMINAL_FUEL_KG = 75.0
DAMAGE_THRESHOLD_J = 209200.0
ARRIVAL_RADIUS_M = 50.0


def score_estimate(time_s, fuel_kg, damage_J=0.0):
    """Winning formula ONLY; callers must independently establish arrival."""
    if time_s <= 0:
        return None
    return float(50.0 + 100.0 * (NOMINAL_TIME_S / time_s
        - 2.0 * damage_J / DAMAGE_THRESHOLD_J
        - max(fuel_kg / NOMINAL_FUEL_KG, 0.0)))


def native_impulse_fuel(jet_force_N, allocation_s, actual_native_dt_s):
    """Exact supplied native pulse formula, given its actual allocation and dt."""
    lam = np.maximum(np.asarray(allocation_s, float), 0.0)
    peak = float(lam.max(initial=0.0))
    if peak < 1e-12:
        return 0.0
    on_time = lam * min(actual_native_dt_s, peak) / peak
    return float(np.asarray(jet_force_N) @ on_time / (ISP_S * G0_M_S2))


def model_fuel_rate(model, public_request):
    """Same pulse law averaged using the identified mean period; estimate only."""
    b = model.allocator.clip(public_request)
    lam = model.allocator.solve(b)
    return native_impulse_fuel(model.allocator.forces, lam, model.h) / model.h
