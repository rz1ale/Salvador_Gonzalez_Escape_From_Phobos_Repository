NONLINEAR HANDOFF STAGING | v1.0.0
Read CONTROLLER_SPEC.txt for the exact model, cost, constraints and control law.

INSTALL
Extract ZIP CONTENTS into your existing STU4-Solution root. This adds
policy/nonlinear_handoff_staging and updates AI_readme_simulation_workflow.txt.
Requires root nominal_model.json, existing system_identification/model.py, and the
shared runner. Does not replace Controller.py/OriginalController.py or native files.
No dependency on naive_handoff, trajectory_handoff, or any old solution clone.

POWERSHELL SETUP (each new terminal)
  $root = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Assets\Repos\STU4-Solution'
  $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
  Set-Location -LiteralPath $root
Use your actual installed STPro Python path if its version differs. Bare python
may invoke the Windows Store alias. $py is a terminal variable, not a Python server.

ONE PREPARATION COMMAND
Close the previous SpaceCRAFT simulation window. Then:
  & $py .\policy\nonlinear_handoff_staging\precompute.py --install-solver --select
  if ($LASTEXITCODE -ne 0) { throw 'Preparation failed; inspect the printed error/report before launch' }
--install-solver installs CasADi==3.7.2 into THIS policy's _offline_packages only
if unavailable. It uses --no-deps; it does not upgrade native NumPy/SciPy or edit
the SDK. CasADi/IPOPT is used OFFLINE ONLY. Internet is needed for a missing wheel.
Subsequent preparations can omit --install-solver.
An unchanged prepared cache is reused immediately. Design changes trigger a new
optimization/replay/gain build; feedback-only changes replay the same reference
and rebuild its gains. The command then selects/checks the existing runner.
It NEVER launches, restarts, or edits a native simulation.
To force another optimization with unchanged settings, add --force.
The shipped cache can be inspected without CasADi:
  & $py .\policy\nonlinear_handoff_staging\precompute.py --check

MANUAL RUN
After successful preparation, open STPro -> unchanged official
STU4-Sim/Sims/COMPETITION_SIM.simconf -> spectator.
Each GUI opening creates an independent recorded attempt. Never terminal-execute
Controller.py or entry.py. The initial command belongs to the optimized trajectory;
there is no prescribed +Z boost or separate rotation stage.
Default execution.test_duration_s=null continues through the maneuver and full-X
continuation until observed arrival or operator stop. A numeric test duration
sends zero at that time and leaves the native sim connected.
Close/end the simulation in the SpaceCRAFT GUI; terminal alternative:
  & $py .\run_simulation.py stop --now
The latter is the shared runner's explicit operator-requested native shutdown.
An error 10054 alone does not establish a Compute_Server crash; inspect result.json.

ONE TUNING FILE: settings.json
design.distance_m                  soft xd reference distance, default 5 m
design.duration_s / steps           fixed maneuver time/grid, default 14 s / 28
design.Q / Q_final                 named position, velocity, attitude, rate blocks
design.R                           six normalized-public-request quadratic weights
design.forward_velocity_m_s        null => FREE velocity; no implicit stop objective
design.terminal_pointing_fraction   default .02 of requested turn (~1.87 deg)
design.terminal_pointing_deg        null => fraction; set absolute degrees to override
design.terminal_rate_rad_s          absolute .005 rad/s (percentage of zero undefined)
design.terminal_position_tolerance_m null => no hard xd distance condition
feedback                           cached tracking period/limits; post-endpoint damping
feedback.position_weight_multiplier default .01; tracking Q/Q_final position only
feedback.rate_weight_multiplier     default 100; tracking Q/Q_final rates only
solver.max_wall_s / max_iterations  offline computational budget; not native sim limits
solver.accept_verified_candidate   true => accept replay-verified nonstationary result
execution.label/status_period_s/test_duration_s  runtime-only, no regeneration needed

Change design/feedback/solver settings while the native simulation is closed, then:
  & $py .\policy\nonlinear_handoff_staging\precompute.py --select
  if ($LASTEXITCODE -ne 0) { throw 'Preparation failed' }
Do not launch after a failed preparation. Runtime rejects stale model/design/source
assets BEFORE sending commands, with an explicit preparation command in its error.
Changing only execution fields does not invalidate the trajectory.
The two feedback multipliers alter the trajectory tracker's weights, not the
offline objective. This explicitly separates planning from tracking tuning.
The program never silently extends duration or replaces endpoint requirements with
soft penalties. d is deliberately soft; pointing/rate bounds govern acceptance.

RESULT STATUS
assets/prepare_report.txt and manifest.json distinguish:
  converged_local_solution: optimizer convergence plus canonical replay passed.
  replay_verified_nonstationary_candidate: endpoint/replay passed; no optimum claim.
The selected iterate is also reported (solver final, feasible iterate, or seed).
These labels do not establish native execution. Relaxed state accuracy does not
mean the objective is within 2% of an unknown optimum.
Current defaults/ship-time measurements are recorded in VALIDATION.txt.
assets/prediction.png shows position error, pointing, rates, inputs, and model fuel.
There are no obstacles/terrain/contact predictions in this optimistic preparation.

FUEL / INPUTS
Public requests are not delivered accelerations. The nonlinear predictor retains
the embedded minimum-sum allocator, shared normalization, gains, gravity, offsets
and delays. Canonical request coordinates improve numerical conditioning without
creating extra thrust; original-model replay is mandatory before export.
The last command's small canonical +X request produces maximum MODEL pure +X
delivery. Continuation common-scales toward native request caps. Rotation shares
allocation, so maximum forward delivery plus arbitrary angular authority is not
promised. Official fuel Isp=80 s and g0=9.8 remain fixed in accounting.py.
R is quadratic effort, NOT exact fuel. Model fuel estimates and measured public
Stats/FuelUsed are kept separate; the latter is primary for run comparison.

AFTER RUNS
  $session = (& $py .\run_simulation.py status | ConvertFrom-Json).session
  & $py .\policy\nonlinear_handoff_staging\analyze.py --session "$session"
  & $py .\policy\nonlinear_handoff_staging\analyze.py --all --show
Logs: policy/nonlinear_handoff_staging/run_logs/<session>/trials/<trial>/<attempt>/
staging_telemetry.txt, staging_summary.txt, settings_used.json, reference_used.json,
plus runner states.txt/commands.txt/events.txt/result.json.
Analysis: policy/nonlinear_handoff_staging/analysis_results/<session>/<trial>/<attempt>/
metrics.txt and run_metrics.png; --all rebuilds one comparison.txt (tab-delimited)
and comparison_details.json. --show also plots the cross-run handoff comparison.
Reports actual handoff pointing/rate/xd error/forward velocity/fuel, source model
predictions, startup delay, sample gaps, damage and whether measured bounds passed.
Incomplete attempts are retained and receive no winning-score estimate.
Official winning formula only after observed arrival, with native scorer authoritative:
  S=50+100*(120/t - 2*damage_J/209200 - max(fuel_kg/75,0)).

OFFLINE CHECKS (after installing solver)
  & $py -m unittest discover -s policy/nonlinear_handoff_staging/tests -v
No native simulation starts. Native qualification remains a separate user test.

NUMERICAL REFERENCES
https://underactuated.mit.edu/trajopt.html
https://web.casadi.org/docs/
https://coin-or.github.io/Ipopt/OPTIONS.html
