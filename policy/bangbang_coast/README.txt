BANGBANG_COAST — separate score/QR spacecraft control policy
============================================================
Status: offline-tested candidate only; NO native qualification or safety guarantee.

Purpose
-------
Optional initial +Z lift -> body +X attitude bang-bang slew -> high-request +X
burn with body Y/Z radar-avoidance corrections -> coast with optional small trim.

The main planner searches a fixed aim direction and one burn time using a fast,
vectorized translation/gravity approximation. The online controller uses the
existing 13-state model, current public state and radar, and the existing
shared runner. It NEVER changes Controller.py, sim_core, STU4-Sim, or any .simconf.

Where to install
----------------
Extract the ZIP into your existing STU4-Solution ROOT, preserving its path:
  STU4-Solution/policy/bangbang_coast/
Files: entry.py, flight.py, core.py, settings.json, run_plan.json,
       analyze.py, tests/test_policy.py, README.txt, __init__.py.

Existing dependencies in your root (NOT included/overwritten):
  Controller.py                 protected bootstrap + shared dispatch
  sim_core/{runtime,common,...}.py
  run_simulation.py
  system_identification/model.py
  nominal_model.json

Python requires NumPy and SciPy, including linprog(method='revised simplex').
The exact allocator LP solver matters because degenerate LP solutions can
change delivered impulses and fuel. The model's nominal native period is
FITTED 0.021 s; it is not an observed physics tick or verified 1/60 s.

Before any native use: run the offline checks
---------------------------------------------
From your STU4-Solution root in PowerShell, with $py set to STPro's bundled
Python interpreter (see AI_readme_simulation_workflow.txt):
  & $py -m unittest discover -s policy/bangbang_coast/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'Bangbang offline tests failed' }
  & $py .\run_simulation.py check

Manual selection (does NOT launch simulator):
  & $py .\run_simulation.py select --config 'policy/bangbang_coast/run_plan.json'
  & $py .\run_simulation.py check

Then open ONLY the unchanged official competition simulation from STPro,
and select spectator as normal. Close the SpaceCRAFT simulation window when
finished. The new manual policy does not automatically launch, reset, or stop
another simulation; a 'success' result is OBSERVED arrival, not the official
scorer's adjudication.

To restore the prior policy, use your saved prior config/selection explicitly;
select --config modifies active_run.json and does not automatically restore it.
For the historical state, the previously selected entry was
policy/nonlinear_handoff_staging/entry.py. Do not overwrite that policy.

Stop/pause from your normal terminal (same $py):
  & $py .\run_simulation.py status
  & $py .\run_simulation.py stop --now
The runner asks the active task to stop, then requests native shutdown.
If manual shutdown is blocked, close the SpaceCRAFT GUI window.
Zero thrust is not braking and cannot guarantee obstacle avoidance.

Tuning
------
Edit ONLY policy/bangbang_coast/settings.json (then select a fresh run again).
- objective.mode = 'score' or 'quadratic' (not 'QR' literal)
- objective.score = official score constants from supplied Nov 2026 simconf
  score estimate = 50+100*(120/T - fuel/75 - 2*damage/209200)
  Actual native scoreboard is authoritative. Predicted future damage is ZERO.
- objective.quadratic.Q_time, Q_terminal_velocity, R_fuel, R_turn choose
  cost among candidate arriving trajectories; Q_position/Q_velocity/R_correction
  parameterize a 6-state local continuous-time LQR used NEAR target only.
  Q/R values are tunable, not physical fuel units.
- planning.burn_min_s/burn_max_s, cone_half_angle_deg, burn_samples,
  grid_refinement_levels, horizon_s, arrival_radius_m set the low-dimensional
  arrival search. 'require_feasible_plan'=true prevents a blind +X launch.
- launch.enabled, duration_s and body_request_m_s2: optional initial +Z boost.
- attitude: bang-bang estimated effective angular acceleration and deadbands.
- burn.request_x_m_s2: raw +X acceleration REQUEST (<=0.5), not delivered accel.
- avoidance: radar-age, relative-speed lookahead, clearance, forward cone,
  body Y/Z lateral request cap; can compete with +X for allocator authority.
- feedback.coast_trim_enabled: enable small nominal coast corrections.
- feedback.qr_feedback_enabled: enable LQR near the target in quadratic mode.
- target.mode: fixed, linear, or angular; velocity/angular rate are estimated
  using previous documented PUBLIC target-position reads, never hidden truth.
- safety: elapsed mission cap, observed-arrival radius, damage and optional
  launch-relative Phobos-center radial-drop guard (NOT terrain/debris clearance).

Scope of model and tests
------------------------
- Uses system_identification/model.py, type public_request_mean_allocation_6dof_v1,
  with the JSON's fitted gains, delay approximations, local Phobos gravity prior,
  jet LP and linear impulse/fuel formula. Current candidate has no independent
  validation flights. The simulator and hardware remain the reference.
- During optimization, pointing is held constant after an ESTIMATED alignment
  transient; target motion is extrapolated; angular fuel/time is approximate;
  no radar obstacles enter the optimized trajectory. During actual flight the
  coupled body-frame allocator, attitude feedback and avoidance alter the path.
- An LP solve or model error intentionally FAILS rather than silently falling
  back to an acceleration-identity model or changing the solver.
- Body Y/Z obstacle avoidance is a radial-TTC heuristic with uncertain, possibly
  stale/noisy radar data. It is NOT a collision-free guarantee; radar lacks full
  obstacle state and does not provide complete scene coverage. No ground-terrain
  avoidance or guaranteed landing/takeoff stability is claimed.
- Quadratic mode uses a continuous-time 6x6 gravity-localized translational A,
  6x3 B at a fixed final pointing orientation for near-target LQR, with the
  nonlinear allocator inverse applied afterward. This is not global linear MPC.
- Arrival means the observed own/target separation <=49 m. The simulator's native
  scoreboard decides actual arrival at its own update time and 50 m threshold.

Offline tests and analysis
--------------------------
  & $py -m unittest discover -s policy/bangbang_coast/tests -v
  & $py .\policy\bangbang_coast\analyze.py --session 'policy/bangbang_coast/run_logs/SESSION_ID' --show
The analyzer writes tab-delimited metrics.txt under:
  policy/bangbang_coast/analysis_results/SESSION_ID/
It never edits raw logs, reads live SDK, or asserts a native scoreboard result.

Recommended first trial: one MANUAL/spectator flight, inspect status/events,
physical position and radar response, then use native score as the reference.
Log optimizer failures and actual test outcomes rather than interpreting an
offline feasible trajectory as a successful flight.
