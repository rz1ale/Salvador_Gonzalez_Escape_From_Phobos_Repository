BANG-BANG COAST V3 — independent exploratory policy
====================================================

Scope
-----
Install under STU4-Solution/policy/bangbang_coast_v3/.
The archive contains NO V2 files, no active_run.json, no Controller.py,
no sim_core/, no native simulator file and no modified model.
V2 Update 2 remains selectable and untouched.

Core idea / differences from V2 Update 2
----------------------------------------
1. The destination is a CAPTURE BALL, not a fixed terminal point or an
   artificial zero-velocity terminal state. Predicted first entry into the
   capture region is interpolated between propagation samples. The planner
   reports predicted_capture_entry_P_m in the initial plan and the current
   predicted_entry_position_P_m in joint forecasts.
2. During main +X burn, V3 periodically searches from the LATEST public
   observed state over (additional forward burn time, lateral YZ direction,
   lateral magnitude, lateral pulse duration). Short, event-triggered
   lateral pulses are interleaved with +X while respecting the TOTAL
   translational request norm cap of 0.5 m/s^2.
3. The mean 38-jet model predicts both delivered thrust and fuel for the
   COMBINED body request. A lateral pulse reduces available +X command.
4. Select a lateral pulse only when the best local plan predicts capture,
   the pulse starts now, and the predicted improvement warrants its fuel
   cost. Never interpret all failed local candidate forecasts as permission
   to burn arbitrarily longer. Radar avoidance has priority over score trim.
5. The initial V2 attitude economy, switching hysteresis, reorientation,
   reactively commanded radar dodges during burn and V2 coast recovery
   remain available; V3 also observes radar passively while coasting.
6. The default coast STILL COMMANDS ZERO LINEAR AND ANGULAR THRUST unless
   a predicted interception miss triggers the V2-compatible recovery.
   Passive coast radar reports DO NOT COMMAND A DODGE. Coast obstacle
   avoidance is NOT implemented or validated in this release.

Inputs and dependencies
-----------------------
Uses the existing public run(context) API and the unchanged
system_identification/model.py + nominal_model.json. Both must exist in
STU4-Solution. Settings and run plan are self-contained in this directory.

Configuration
-------------
Edit policy/bangbang_coast_v3/settings.json while the simulator is CLOSED.
The key settings are:

  objective.mode: score | quadratic
     score: predicted native time/fuel/damage score; quadratic: Q_time,
     R_fuel and R_lateral only. No terminal velocity penalty in either.
  planning.arrival_radius_m: initial nominal capture margin (<50 m).
  burn.joint_optimizer.enabled: true (false => V2 online burn-stop logic).
  burn.joint_optimizer.capture_radius_m: online conservative capture
     radius (42 m default); smaller means more conservative and can cause
     expensive additional burn. This is NOT the official 50 m boundary.
  burn.joint_optimizer.check_period_s: planning cadence (2 s default).
  burn.joint_optimizer.future_burn_window_s: max additional burn explored.
  burn.joint_optimizer.burn_samples: number of burn durations to explore.
  burn.joint_optimizer.lateral_request_magnitudes_m_s2: body YZ amplitudes.
  burn.joint_optimizer.lateral_pulse_durations_s: pulse durations to test.
  burn.joint_optimizer.include_diagonals: include 45-degree YZ directions.
  burn.joint_optimizer.min_score_gain_for_lateral_points: minimum
     predicted improvement before applying a score trim (0.7 default).
     In quadratic mode, the same numerical value is in quadratic cost
     units, not score points; retune before using quadratic mode.
  burn.joint_optimizer.max_lateral_fuel_kg: upper pulse fuel estimate.
  burn.joint_optimizer.max_lateral_episodes_per_burn: avoid repeat trim.
  burn.joint_optimizer.coast_radar_observation: monitor and log only.
  coast.miss_monitor: existing free-intercept coast forecast/recovery.
  coast.direct_recovery: cheapest body-frame correction before turn.
  attitude: existing V2 economical alignment and hold settings.
  avoidance: existing V2 Update 2 threat detector (no rear-axis filter).

IMPORTANT: Controller update, native jet-selection loop period, and physics
integration period are distinct. The fitted nominal model uses an effective
period. No exact native timing or collision clearance is guaranteed.

Installation / selection (PowerShell)
-------------------------------------
1. CLOSE SpaceCRAFT and run_Server before changing the policy selection.
2. Extract the ZIP contents directly into STU4-Solution. Keep its folder
   hierarchy. No existing V2 files should be replaced.
3. From STU4-Solution (use your established STPro $py path):

   & $py -m unittest discover -s policy/bangbang_coast_v3/tests -v
   if ($LASTEXITCODE -ne 0) { throw 'V3 offline tests failed' }
   & $py .\run_simulation.py select --config 'policy/bangbang_coast_v3/run_plan.json'
   if ($LASTEXITCODE -ne 0) { throw 'V3 policy selection failed' }
   & $py .\run_simulation.py check
   if ($LASTEXITCODE -ne 0) { throw 'Runner check failed' }

4. Launch ONLY the organizer-approved unchanged official simulator as in
   the existing workflow documentation; select spectator.
5. For another trial, close GUI, reselect V3 if settings changed, then
   start a NEW manual launch. Do not edit the simulator configuration.

Logs
----
policy/bangbang_coast_v3/run_logs/... includes public state, six-axis
commands, public radar, events, hashes and results from the shared runner.
New events: joint_burn_forecast, joint_lateral_started,
joint_lateral_completed, joint_lateral_cancelled_for_radar,
coast_radar_risk_observed. Nominal plan and post-align refit events include
predicted capture-entry locations; never assume a fixed x_f.
The legacy V2 analysis helper is copied for convenience, but primary
performance evidence remains the shared recorded session and native scorer.

Limitations / decision points
----------------------------
- The offline validation is NOT native simulator validation. New score
  decisions are based on a fitted approximate model; a forecast can be
  wrong due to attitude hold, radar dodges, timing, gravitation and reports.
- This optimizes a PARTICULAR low-dimensional bang/off/pulse schedule,
  not global 6-DOF optimality or stochastic collision probability.
- Candidate prediction assumes the current attitude remains fixed through
  the near-term burn, with zero future angular corrections; the actual
  controller may command angular hold bursts and reactive obstacle dodges.
- Radar-based collision prevention is reactive during burn/align; coast
  radar is passive analysis, not collision avoidance. Unknown debris
  tangential velocity prevents a rigorous clearance guarantee.
- No terminal translational speed constraint or penalty exists.
- Alternative Q/R objective uses quadratic time/fuel/lateral cost and
  optional LQR corrections ONLY if enabling corrected coast mode.
- Only competitor source and public observations may be used at runtime;
  protect original Controller.py and follow the official rules.

Suggested first experiment
--------------------------
One native manual score-mode flight. Compare initial alignment fuel,
main-burn duration/fuel, joint_lateral_started count, unexpected coast
recovery, recorded score, distance and damage with the two V2 Update 2
benchmarks: 47.2 (217.56 s, 43.46 kg) and 47.7 (216.16 s, 43.37 kg).
DO NOT replace the V2 submission with V3 without independent native tests.
