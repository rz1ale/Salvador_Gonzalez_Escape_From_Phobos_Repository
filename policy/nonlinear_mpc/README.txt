NONLINEAR MPC | v0.1.0 | 2026-11-07 | FIRST DIRECT-SHOOTING BASELINE
Read ROOT/AI_readme_competition_overview.txt and AI_readme_simulation_workflow.txt.
Evidence labels: OFFLINE_MODEL_RESULT / SCORE_ESTIMATE / native official result.

LAYOUT / OWNERSHIP
  policy/nonlinear_mpc/entry.py       sole run(context) entry, both launch modes
  optimizer.py                       pure coupled numerical problem/predictor
  worker.py                          one owned numeric subprocess; no SDK
  check.py                           read-only offline preflight
  settings.json                      canonical defaults; case overrides permitted
  run_plan.json / smoke_run.json      shared RUNNER plans, never native sim configs
  analyze.py                         local report/plots; supports --analysis hook
  tests/test_policy.py               offline software/model checks
  run_logs/SESSION/trials/...         existing runner attempt organization
  analysis_results/SESSION/          metrics.txt, metrics.json, run_metrics.png
No Controller copies, private simulator launcher, restarts, old-clone imports,
native changes, alternate .simconf, or duplicate plant backend are supplied.
Keep variants as named run-plan cases with parameters.overrides; reuse this entry.
Different future formulations belong behind optimizer dispatch, not copied trees.

DEPENDENCIES / INSTALL
Extract this ZIP's CONTENTS into the existing STU4-Solution root. It adds only
policy/nonlinear_mpc/. Existing shared runner, root nominal_model.json,
system_identification/model.py and obstacle_characterization/ must already exist.
Uses their current public contracts; no shared framework files are overwritten.
The selected model must be public_request_mean_allocation_6dof_v1, schema 1,
embedding the 38-jet matrix, positive forces, mass, gains, delays and offsets.
It is loaded via context.model_path; run plans select ROOT/nominal_model.json.
Historical absolute paths inside JSON provenance are metadata, never dependencies.
Backend/source compatibility pins and static mass/inertia/caps/jet geometry are
checked; organizer updates require review,
not automatic pin replacement. Static official files are read only, never imported.
These checks cannot authenticate the full organizer asset or the manual GUI choice;
the operator must select the unchanged official competition simulation.
Runtime imports: NumPy/SciPy already used by identification; Matplotlib for reports.

NUMERICAL PROBLEM
Direct single shooting, SciPy SLSQP, finite-difference derivatives, warm starts.
18 normalized decisions = 3 piecewise-constant stages x 6 simultaneous RAW public
requests. Scaling uses positive/negative allocator saturation knees, not native
request caps mistaken for achieved acceleration. Joint allocation is evaluated.
State: effective-point p,v, passive q_PB xyzw, omega_B. All 6 inputs are optimized
together through translation/attitude dynamics and shared allocation/fuel cost.
No separate attitude tracking or translation controller supplies the main policy.
Canonical Model._advance supplies dynamics. Policy midpoint mesh is 0.15 s;
comparison with canonical 0.01 s integration is an offline check, not native proof.
Timestamped command history preserves 0.025/0.020 s effect delays; initialization
uses Model.causal_initial for reporting offsets/delays. No raw request=accel shortcut.

Defaults: horizon 6 s, stages 2 s, path knots 0.5 s, replanning about 0.33 Hz,
solve budget 2 wall s, planned activation 3 simulation s after snapshot. SDK
monitor/command refresh targets 0.2 wall s; actual I/O/work adds overhead.
One numeric-only subprocess isolates the solve. SDK calls stay on controller thread.
Committed commands are propagated through the activation lead. Late/generation-
mismatched results are rejected. Fresh full-object checks gate accepted plans;
state/attitude/rate deviations invalidate execution. Plans expire explicitly.
SLSQP may return a feasible incumbent without convergence. No global optimum or
hard real-time feasibility is claimed. A 4 wall s worker watchdog stops a stuck
attempt and cleans only that owned child; no simulator process is spawned here.
The child also exits after 10 wall s without its owner's pulse after abrupt GUI
shutdown; graceful entry cleanup closes it immediately.

SCORING OBJECTIVE
Supplied official scorer: S=50+100*(120/t - 2*damage_J/209200 - fuel_kg/75).
Constants are read from the unchanged official competition file each attempt;
reviewed Scoring.py and JetSelectionActuator.py content is pinned (LF normalized).
Native success: target distance <=50 m, unless damage previously exceeded limit.
No terminal velocity/attitude mission condition or 75 kg physical fuel cap is added.

Objective: maximize predicted completion S, minus small documented rate/slew
regularizers, subject to local collision, rate and launch-plane constraints.
Because the mission exceeds the radar/model-local horizon, completion time/fuel
uses an explicit obstacle-free accelerate/coast tail estimate. Its thrust plateau,
impulse price, gravity-support fuel and lateral/rate correction depend on terminal
state/attitude and the SAME embedded allocator. tail_fuel_multiplier is tunable.
A terminal rate bound (0.01 rad/s) makes the held-attitude tail consistent; it is
an optimization constraint, not an organizer arrival requirement. Forecasts beyond
the local radar horizon are NOT declared collision-free. Tail cost is approximate;
receding-horizon optimization of it is not exact full-mission score optimization.
Fuel prediction sums force-weighted LP activity / max(h,peak) / (80*9.8).
h is the identified mean-allocation scale, not a claimed native live tick.

OBSTACLES / RECOVERY
Consumes the existing RadarStream, no debris truth/camera/private native data.
Uses ALL supplied tracks, tentative/coasting/stale objects and unresolved hazards;
no arbitrary top-N truncation. CV spheres include shared uncertainty/shape/clock
inflation, plus vehicle radius 4 m, extra clearance 3 m and chord padding 0.1 m.
Swept relative closest approach is checked over whole segments, not just knots.
4 m about the SDK reported origin is an ASSUMPTION, not a verified enclosing hull.
Native VisualRadius_m=3 does not establish collision geometry. Confirm/tune the
policy radius from permitted vehicle geometry; never alter the native vehicle.
Chord padding and statistical spheres are not a physical continuous-safety proof.
The local launch-plane floor comes from the first own-position radial direction;
it is not a terrain map or a guarantee against Phobos surface collision.
Missing/partial coverage is logged, never certified as free space. Capacity loss,
unavailable own pose and future-stamped data block trajectory acceptance.
When no usable plan exists, bounded six-axis recovery seeds (brake/hover/progress,
then simple escape directions) are screened by the same nonlinear local check.
This recovery is not optimal; zero thrust is not braking. No admissible candidate,
prolonged recovery, timeout, worker failure or official damage threshold -> recorded
task failure and normal stop. Failures stay in the benchmark; no success reroll.

POWERSHELL: OFFLINE CHECKS (CLOSE SPACECRAFT BEFORE SELECTION)
  $root = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Assets\Repos\STU4-Solution'
  $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
  $gui = Join-Path $env:PUBLIC 'SimDynamX\STPRO\Apps\Main_SG\SpaceCRAFT\SpaceCRAFT.exe'
  Set-Location -LiteralPath $root
  & $py -m policy.nonlinear_mpc.check
  & $py -m unittest discover -s .\policy\nonlinear_mpc\tests -v
Use the installed STPro interpreter if version/path differs; bare python is not
required. Nonzero exit means resolve the reported issue before selecting flight.
The checks do not launch a native sim. The bundled runner is already installed;
do not reinstall/overwrite Controller or OriginalController for this policy.

FIRST NATIVE SMOKE (PENDING; REQUIRED BEFORE CAMPAIGN)
  & $py -m policy.nonlinear_mpc.check --config 'policy/nonlinear_mpc/smoke_run.json'
  & $py .\run_simulation.py select --config 'policy/nonlinear_mpc/smoke_run.json'
  & $py .\run_simulation.py check
Open Space Teams PRO -> official STU4-Sim/Sims/COMPETITION_SIM.simconf -> spectator.
Same entry/worker/radar/model; 12 s, reduced request bounds (0.35), NONZERO THRUST.
Pass requires an activated MPC plan, nonzero commands, multiple radar frames and
no new damage. Normal result then STOP; manual mode never restarts the simulation.
  & $py .\run_simulation.py status
Inspect result.json for reason=nmpc_smoke_complete and metrics.smoke_passed=true.
Schedule complete alone does NOT establish a pass. To interrupt: in a separate
PowerShell at ROOT with $py initialized, use:
  & $py .\run_simulation.py stop
This requests stop through the shared runner. Do not kill processes by image name.

MANUAL MISSION (AFTER SMOKE PASSES; CLOSE SPACECRAFT FIRST)
  & $py .\run_simulation.py select --config 'policy/nonlinear_mpc/run_plan.json'
  & $py .\run_simulation.py check
Open unchanged official competition sim and spectator manually. Each open creates
a fresh session. No terminal launches/restarts. Observed arrival is 49 m with 0.4 s
dwell to leave native scorer time to sample; native threshold remains 50 m.
Result is observed_arrival, SCORE_ESTIMATE only. The manual sim stays connected,
sends the runner's final zero request, and shows the native scorer alert. Close
the SpaceCRAFT simulation window yourself. Record its official time/fuel/score.
No score is reported as achieved before observed arrival.

AUTOMATIC RUNS (SAME ENTRY / OFFICIAL SIM)
After closing native GUI/backend, and verifying spectator is FIRST role:
  & $py .\run_simulation.py auto --config 'policy/nonlinear_mpc/smoke_run.json' --runs 1 --retries 0 --executable $gui --first-role
Then, after smoke success:
  & $py .\run_simulation.py auto --config 'policy/nonlinear_mpc/run_plan.json' --runs 3 --retries 0 --executable $gui --first-role --analysis 'policy/nonlinear_mpc/analyze.py'
--first-role is boolean; no pilot/spectator argument follows it. Only the shared
supervisor launches/stops/restarts official sim processes. Manual never restarts.
For variant sets, add cases to run_plan.schedule.cases, e.g.:
  {"id":"H4","parameters":{"overrides":{"horizon_s":4.0}}}
  {"id":"MARGIN5","parameters":{"overrides":{"clearance_m":5.0}}}
One manual new_on_open selection must contain one case; use --case H4 to select it.
Automatic mode can execute the set. Shared shallow per-case merge applies;
an overrides object replaces the global overrides object, then overlays defaults.
Retry infrastructure failures only, retaining all attempts; task failures advance
as recorded failures according to the runner plan. Do not edit a running session.

LOCAL REPORT / DIAGNOSTICS
  $s = (& $py .\run_simulation.py status | ConvertFrom-Json).session
  $rel = (Resolve-Path -LiteralPath $s -Relative).Replace('\','/')
  & $py .\policy\nonlinear_mpc\analyze.py --session $rel --show
Run from ROOT, or copy the solution-relative session path from status, e.g.
policy/nonlinear_mpc/run_logs/SESSION.
Existing attempt logs: states.txt, commands.txt, radar.txt, events.txt, result.json.
Added: policy_telemetry.txt (guard/recovery/counters), solver_results.txt (plans),
policy_settings.json, obstacle_diagnostics.txt, numeric_worker/worker.txt plus
one latest request/response. No copied controller trees or data snapshots.
Reports include solve timing, recovery vs MPC usage and public observed fuel vs
replayed predicted fuel. Changed model hashes disable fuel replay, not saved data.
Before performance tuning, inspect activated_plans, rejected_results, control work
time, collisions/guards and fuel ratio; a mostly-recovery run is not MPC evidence.

VERIFICATION / LIMITS
15 offline checks passed, including real numeric-process handshake, complete
run(context) with public-interface mathematical fixture, cleanup, timing/history,
canonical prediction comparison, shared plan validation and infeasible-obstacle
rejection. Read validation.txt for the release evidence and remaining limitations.
These tests are not a native smoke pass, validated debris geometry, global
optimality, native score, or a 60+ score guarantee. Run native smoke first.
