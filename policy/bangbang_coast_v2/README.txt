BANGBANG COAST V2 — UPDATE 2 (2026-11-18)
===================================================
Status: OFFLINE TESTED; Update 2 NOT VALIDATED IN NATIVE SIMULATOR.

This is a NEW policy, not a replacement for policy/bangbang_coast (v1).
Unzip at the STU4-Solution root. The ZIP contains ONLY:
  policy/bangbang_coast_v2/*
It does not modify Controller.py, run_simulation.py, sim_core, STU4-Sim,
original policies, active_run.json, or official .simconf files.

UPDATE 2: ATTITUDE CHATTER + REAR-HEMISPHERE RADAR THREATS
---------------------------------------------------------
This targeted patch changes only policy/bangbang_coast_v2 source/settings/tests.
V1 is never changed. The prior successful nominal interception, ballistic coast,
online burn timing and reinterception logic are retained.

- Initial alignment has a stateful switching guard. A requested angular torque
  reversal within minimum_reverse_dwell_s after the preceding reversal produces
  a zero command until the delay expires, not a prolonged incorrect torque.
  Angular switching counts and inhibited reversals are logged at alignment end.
  This is a chatter guard, NOT an optimal attitude-control proof. The policy
  still aims to settle perpendicular angular velocity; translational terminal
  velocity is NOT driven to zero in score mode.
- Radar avoidance considers nearby closing objects at any body-X coordinate,
  including behind the shuttle. It uses public measured radial relative speed,
  shuttle inertial velocity, approximate relative-center kinematics and a
  heuristic future closest approach. It does not use debris truth, unique IDs,
  or infer unmeasured true obstacle tangential velocity.
- A predicted close encounter selects a body-frame Y/Z dodge; imminent threats
  may use the separately configured emergency request magnitude (<=0.5).
  Shared actuator coupling can lower delivered body-X thrust. Multiple threats
  are handled by a prioritized heuristic, not a collision-free path planner.
- Public impact-flight regression: the previously discarded closing object at
  sim 35.146 s (22.6 m surface range, -4.15 m/s radar radial velocity, body-X
  approximately -21 m) is now flagged and an emergency dodge is requested.
  This does NOT prove the dodge would avoid the observed collision in native.

Settings:
  attitude.alignment_control.minimum_reverse_dwell_s     0.45
  attitude.alignment_control.opposing_direction_cosine   -0.35
  avoidance.min_closing_speed_m_s                         0.25
  avoidance.radar_age_uncertainty_m_s                    1.0
  avoidance.emergency_time_to_threat_s                   6.0
  avoidance.emergency_surface_range_m                    35.0
  avoidance.emergency_lateral_request_m_s2               0.4
  avoidance.max_lateral_request_m_s2                     0.15
  avoidance.lookahead_s                                  8.0

The old forward-only filter (`min_forward_cosine`) and body-X-based lateral
range gate were intentionally removed. Radar is still NOT queried during
ballistic coast, which can encounter unobserved debris. Mitigating that requires
an additional event-triggered coast avoidance/replanning design.

Control architecture
--------------------
1) Optional +Z launch boost (configurable).
2) Search nominal heading and +X burn duration with nominal allocator fuel
   and either official-score estimate or user-configured quadratic weights.
3) Closed-loop bang-bang-like attitude alignment. A low-dimensional economy
   choice compares angular request magnitudes using delivered angular authority,
   fuel, and estimated turn time. Degenerate identical responses do not imply
   cheaper fuel when the request magnitude is lower.
4) Body +X powered flight with body Y/Z radar correction. The controller keeps
   the combined translational REQUEST norm <= 0.5 m/s^2. Angular commands
   are independent requests capped at 0.05 rad/s^2, but all six DOF share the
   native jet allocator, so attainable acceleration is coupled.
5) Event-triggered attitude holds: zero angular request when pointing/rate
   are acceptable; bounded correction bursts when drift exceeds thresholds.
6) Reevaluate coast-now versus further +X burn at intervals using the observed
   state, approximate nonlinear gravity, and nominal jet effectiveness.
7) Ballistic coast: repeatedly send zero commands, no attitude/lateral/trim
   thrust. Forecast future closest approach from current public state.
8) On confirmed miss, rank short BODY-FRAME translation pulses (primarily Y/Z),
   estimate fuel through the 38-jet allocator, and verify their trajectories
   using the nonlinear mean-state model. Only if direct correction fails do we
   consider a full attitude reorientation and a new +X burn.
9) After a direct pulse, stop all commands and re-evaluate the ballistic coast.
   Recovery attempts are bounded and logged; zero thrust is the default.
10) The initial alignment uses a vector stopping-distance switching surface
    with conservative braking and no unnecessary body-X roll correction.
11) Burn duration cannot silently follow the original schedule when the latest
    forecast is infeasible: additional burn is permitted only if it has a
    feasible high-score projection or meaningful miss reduction per kilogram.

IMPORTANT: The coast miss monitor is NOT a full radar obstacle predictor and
cannot guarantee avoidance or success. During ballistic coast it does not
command radar-triggered dodges. A coast prediction relies on the fitted local
13-state model, reporting-offset convention, local gravity prior and an
approximate integration step; it is not simulator truth.

INSTALL AND SELECT
------------------
From STU4-Solution in PowerShell, with $py pointing to STPro's bundled Python:
  & $py -m unittest discover -s policy/bangbang_coast_v2/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'V2 offline tests failed' }
  & $py .\run_simulation.py select --config 'policy/bangbang_coast_v2/run_plan.json'
  & $py .\run_simulation.py check

Now open ONLY the unchanged official simulator in STPro, select spectator,
and run the selected manual policy. The code does not start or alter a sim.
The active_run.json selection is updated by run_simulation.py; the prior
selection can be restored with:
  & $py .\run_simulation.py select --config 'policy/bangbang_coast/run_plan.json'

Do not run Controller.py manually. To request stop during a manual session:
  & $py .\run_simulation.py stop --now
If the manual controller is blocked, close the SpaceCRAFT GUI window.
Manual execution leaves the simulator connected until it is closed or stopped.

CONFIGURATION: policy/bangbang_coast_v2/settings.json
------------------------------------------------------
objective.mode: 'score' (default), or 'quadratic'. Quadratic uses Q_time,
Q_terminal_velocity, R_fuel and R_turn in the trajectory search. An optional
6-state local translational LQR is used in corrected coast ONLY; there is NO
full 13-state MPC. Set coast.mode='corrected' to enable the old corrected
coast behavior, which intentionally uses propellant.

attitude.economy.*: candidate angular request magnitudes, minimum effective
angular acceleration, and turn/fuel cost multipliers.
attitude.alignment_control.*: conservative braking acceleration fraction,
quiet switch region and estimated attainable stopping distance. These are
controller-design heuristics; do not treat gains as physical constants.
attitude.burn_hold.*: correction activation/release hysteresis, angular rate
thresholds, correction burst duration and maximum request. In emergency the
angular hold may stay active longer to recover pointing.
burn.online_stop.*: score/feasibility lookahead and permitted burn extension.
It evaluates models with future pure +X; actual angular/lateral corrections
may change the result, so observed-state re-evaluation matters.
coast.mode='ballistic' retains zero-command coasting, except when the
miss monitor explicitly selects a recovery maneuver.
coast.miss_monitor.*:
  enabled                 true by default for ballistic mode
  entry_grace_s           wait before the first predicted-miss check
  check_period_s          public-state prediction cadence in sim seconds
  integration_step_s      numerical prediction step (not native physics tick)
  horizon_s               maximum future prediction horizon
  capture_radius_m        conservative modeled capture radius (43 m)
  trigger_closest_m       trigger if repeated closest approach >48 m
  confirmations_required  number of consecutive predicted-miss checks (2)
  max_reinterceptions     total direct-pulse or full-turn recoveries allowed (2)
  attempt_cooldown_s      minimum time between planning attempts
The monitor does NOT claim safety if its forecast fails. A model whose path is
still closing at the end of a short horizon is treated as unresolved rather
than automatically as a miss, unless the forecast covers the remaining
mission timeout. The predictor intentionally includes the body-fixed reported
point offset as attitude changes. A feasible replan is required BEFORE any
recovery thrust; if none exists, zero-command coast continues and logs why.

coast.direct_recovery.*:
  enabled                    allow short body-frame correction pulses
  arrival_radius_m           conservative predicted capture target for long coast
  urgent_arrival_radius_m    relaxed margin if time-to-closest is very short
  pulse_min_s / pulse_max_s  bounded pulse duration search
  request_magnitudes_m_s2    candidate request magnitudes (<= 0.5)
  max_detailed_candidates    number of candidates verified with model.step()
  max_fallback_turn_deg      limit on full-reorientation fallback (180 default)
  min_fallback_predicted_score  avoid negative-score fallback only if desired
Any pulse is followed by zero commands and a fresh forecast. The pulse search
may block briefly; commands are zero before planning. Delays, debris risk, and
model errors remain and are not certified away by the nominal simulation.

PROVENANCE AND LIMITS
---------------------
Uses system_identification/model.py and nominal_model.json supplied in the
existing STU4-Solution root. They contain the 38-jet mean allocation model,
fitted 0.021 s period, delays, estimated reporting offset and gravity prior.
These fitted values are NOT direct measurements of native physics timing.
LP minimum-sum allocations can depend on solver tie-breaking; the model uses
the declared SciPy revised simplex implementation. Known model candidate has
no independent validation flights, so synthetic success cannot predict native
score, failure probability, or collision clearance.

The scoring optimization is LOW DIMENSIONAL, not a globally optimal solution.
The 50 m native scoring boundary and native score are authoritative. The
controller uses an observed arrival radius of 49 m. Radar data is noisy and
incomplete; body-frame avoidance during the powered phase is heuristic. No
terrain clearance or guaranteed safe rerouting is implied.

LOGS AND CHECKS
---------------
Events: attitude_economy_profile, alignment_complete, online_burn_forecast,
burn_ended (fuel and hold activity), coast_started, coast_intercept_forecast,
coast_direct_correction_selected, body_correction_started,
body_correction_completed, coast_reinterception_trigger, recovery_started,
reinterception_plan,
and any prediction/replanning failure or exhausted recovery budget.

To produce a read-only summary from an existing run:
  & $py .\policy\bangbang_coast_v2\analyze.py --session 'policy/bangbang_coast_v2/run_logs/SESSION_ID' --show

Offline tests include a successful uninterrupted coast, a deliberately
perturbed coast that triggers reinterception, and a synthetic full recovery.
Native simulator use and repeatability remain to be established. Start with
ONE manual flight, compare native time/fuel/score with v1, and retain logs.

V2 CONTROL UPDATE 1 — EVIDENCE / LIMITS
-------------------------------------
Based on the 2026-11-18 19.9-point run: initial alignment 7.66 kg,
recovery alignment 16.00 kg, and 125-degree reorientation. A nominal-model
replay of the measured coast state at sim ~76 s finds a 2.8 s body-frame
Z correction, estimated fuel 1.15 kg and predicted closest approach 33.8 m.
These are offline model results, not an observed native improvement.
The initial alignment law also reduced nominal model turn fuel versus its
previous control law in a matched-state comparison, but native results are
not yet measured. Always compare actual score/fuel on separate runs.

UPDATE INSTALL
--------------
Extract the patch ZIP into STU4-Solution and replace V2 files.
V1 files are untouched. Close SpaceCRAFT first, then run offline tests
and reselect the V2 run_plan.json using run_simulation.py select --config.
A paused/active session with prior source hashes cannot be safely resumed
with patched code; create a new session.
