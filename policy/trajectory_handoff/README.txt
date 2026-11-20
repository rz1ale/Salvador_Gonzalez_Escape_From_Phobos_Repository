TRAJECTORY HANDOFF POLICY | v0.2.0 | 2026-11-15 (America/Chicago)

PURPOSE / LEGAL SCOPE
Use the existing shared runner, root Controller.py and unchanged official
STU4-Sim/Sims/COMPETITION_SIM.simconf. Only this solution policy is updated.
No historical C-series launcher, .simconf, native actuator/radar/scorer wrapper,
Mars/SSB entity reference, hidden debris state, seed or live allocator telemetry.
Inputs: permitted own state, target, public radar, selected nominal_model.json.
SDK calls remain on the controller thread. Children read public logs only.
Includes the complete shared obstacle_characterization v0.1.1, byte-identical to
the previous release. Runner, canonical model backend and cache are unchanged.

EXECUTION CONTRACT
1. Start cached coupled FF+FB immediately, before constructing numeric workers.
2. Independent monitor: 0.5 s request period, 0.8 s soft budget, 1.2 s activation
   lead, 10 s local forecast. Track/update actual current plus pending commands.
3. Mission planner: normally 4 s request period, 3 s end-to-end soft budget,
   4 s activation lead. A new local hazard/commitment can request earlier work,
   at most once per second. Advisory warnings alone do not force this faster rate.
4. Both use one persistent numeric child EACH, with BLAS threads=1. This deliberate
   two-child design prevents a slow mission solve from blocking the monitor.
   Both start after the first command; neither owns native launch/restart.
5. Control/own-state loop remains 0.1 s; public radar polling remains 0.4 s.
   Actual rates depend on SDK latency, CPU and OS scheduling; these are targets.

CONTINUATION / HANDOVER
There is no no_fresh_verified_control abort and no plan-horizon abort.
Failed, late or infeasible mission replanning leaves current control available.
The old assessment lifetime is now diagnostic, not a native-stop authorization.
At a plan's executable endpoint, select explicit zero-translation coast with
bounded angular feedback, while monitor/planner continue. This does not hover,
stop velocity, cancel gravity, or establish geometric clearance. Forecast coast
attitude is taken at the NEAR executable endpoint, never the coarse mission-tail
endpoint; this matches the controller's explicit coast continuation.
Tracking departures also retire the plan into explicit coast and request repair.
Stale/unavailable monitor data after startup selects this explicit coast, not
an indefinitely held launch burn. Stale own state, invalid commands, unacceptable
send latency, public-target mismatch and official damage threshold still fail.
Worker death/timeout is logged, disables that numeric role, and does not itself
end the simulation. The remaining role and fast control continue; cleanup owns
both children. No automatic native retries are added.

Local avoidance samples finite 1/2/3 s body-translation pulses followed by coast,
using the current six-axis allocator, full nonlinear dynamics and angular
feedback. It rechecks the current commitment each monitor update. A still-clear
commitment persists; failed optimization does not resurrect a retired nominal
path. A 0.5 m improvement hysteresis discourages switching among equally bad
sampled continuations; it is NOT a reduction of geometric margins.
A sampled continuation which passes every configured query is labeled separately
from guard_risk_fallback. If no candidate passes, continue the best available
sampled/current continuation; record risk and failure conditions explicitly.
This is intentionally allowed to collide. No assertion of collision probability,
recursive feasibility, hull enclosure or formal safety is made.

New plans require generation, receipt, timing, activation-state and tracking
checks. Monitor proposals cannot cancel an earlier pending switch on which their
predicted bridge depends. Pending switches appear in both workers' forecasts.
New local commitments invalidate obsolete mission proposals. Rejoin to the
cached reference remains an explicitly checked action with two clear votes.

PLANNING / SCORE
Keep current coupled 13-state quaternion model and six-axis saturation/fuel map.
Default correction variables remain three body-translation requests; angular
feedforward and full-state feedback continue. correction_axes=all is optional.
Do NOT substitute the old C-series acceleration-identity/fixed-attitude model.

The near grid remains ten 1.5 s stages (15 s). Eight additional nonuniform stages
cover an advisory region and a coarse continuation to the mission target.
Arrival-time scales 1.0/0.9/1.1 are attempted within budget. The affine terminal
constraint is an inscribed 48 m target box; terminal velocity is not constrained.
Frozen continuous error Jacobians include gravity, attitude/translation coupling,
gyroscopic terms, allocator saturation derivatives and separate channel delays.
Their exact affine ZOH is a planning approximation; short-step comparison to
finite-difference nonlinear transitions showed max entry errors below 1.1e-5 in
three tested states. This is not a long-horizon accuracy bound.
QP decisions reuse warm starts and the allocator's certified unique LP bases.
Up to 12 threatening objects enter a QP working set; nonlinear near-path checks
still assess ALL retained objects before a candidate becomes executable.

Thirty-five-second forecasts are advisory, with full propagated uncertainty.
Only near-path checks gate immediate execution; uncertain distant occupancy adds
route cost and requests replanning. Shared statistical forecast support is 45 s.
A coarse tail is NOT declared collision-free and is NOT executed blindly:
execute_until_sim_s limits the exported executable window to 15 s. Later work
must replace it, or the explicit coast continuation applies.
Target-reaching affine candidates are ranked by current total time/fuel score,
with side-continuity cost. A short verified seed is a fallback if no mission
candidate is found; its cached-tail estimate is not directly interchangeable
with a complete coarse mission prediction. Both approximations are logged.
S = 50 + 100*(120/T - 2*damage_J/209200 - fuel_kg/75).
Native scorer is authoritative; no worst-case/global optimum is established.

MODEL / PERFORMANCE STATUS
The existing offline cache is unchanged: duration 253.9328 s, predicted fuel
34.7494 kg, optimistic score 50.9241, final canonical distance 48.5180 m.
IPOPT hit its iteration limit: feasible incumbent, not proven optimum or 60+.
47 policy checks plus 40 shared tests passed before final packaging, including
actual shared dispatch, both real numeric children, repeated handoffs, independent
monitor results during mission solves, and continued commands through expired
plans/repeated planner failures. Mathematical fixtures do not reproduce native
contacts, CPU contention, SDK behavior or an official competition score.
The three supplied final problem states yield monitor_clear, with local margins
+22.03/+9.27/+10.49 m and advisory warnings. Local public-pipeline times were
0.192/0.153/0.158 s; receipt checks pass on recorded states. Each check starts from
the original flight; no subsequent native flight outcome is inferred.
A moving synthetic public-radar hazard also triggers an actual local handover in
the mathematical integration test; it has no contact model and cannot demonstrate
native obstacle avoidance. The two newly reported early failures have no supplied
logs and remain unclassified.

INSTALL / CHECK
Close SpaceCRAFT and its owned run_Server before overwriting files. Extract ALL
STU4_trajectory_handoff.zip contents into the active STU4-Solution root. Replace
the separately supplied AI_readme_simulation_workflow.txt in the root.
Do not resume an old source-pinned session. No offline cache rebuild is needed. If nominal_model.json changes, the old cache
must not be reused: precompute.py is the existing offline builder, requires its
offline CasADi dependency, and publishes only after canonical replay validation.
Run `& $py -m policy.trajectory_handoff.precompute --help` for its retained options.
New PowerShell window:
  $root = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Assets\Repos\STU4-Solution'
  $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
  Set-Location -LiteralPath $root
  & $py -m policy.trajectory_handoff.check
  if ($LASTEXITCODE -ne 0) { throw 'Handoff preflight failed' }
  & $py -m unittest discover -s policy/trajectory_handoff/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'Policy tests failed' }
  & $py -m unittest discover -s obstacle_characterization/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'Shared tests failed' }
Use the actual installed STPro interpreter if v0.55.0 is absent; no bare python.
Preflight: policy_version=0.2.0, independent_local_monitor=true,
assessment_expiry_stops_sim=false, target_reaching_coarse_tail=true.

MANUAL SMOKE / MISSION
  & $py .\run_simulation.py select --config 'policy/trajectory_handoff/smoke_run.json'
  & $py .\run_simulation.py check
Open Space Teams PRO -> ONLY official COMPETITION_SIM.simconf -> spectator.
Do not execute Controller.py in a terminal. Smoke runs 30 simulation seconds,
forces three actual mission-plan handoffs, and requests normal stop after its
result is committed. A resulting 10054 popup can accompany this intentional stop.
Pass: handoff_smoke_complete, metrics.smoke_passed=true, three takeovers and
monitor results. Full-mission selection (no intentional 30 s smoke stop):
  & $py .\run_simulation.py select --config 'policy/trajectory_handoff/run_plan.json'
  & $py .\run_simulation.py check
Then open the same official file and select spectator.

AUTOMATIC / STATUS / STOP
  $gui = Join-Path $env:PUBLIC 'SimDynamX\STPRO\Apps\Main_SG\SpaceCRAFT\SpaceCRAFT.exe'
  & $py .\run_simulation.py auto --config 'policy/trajectory_handoff/smoke_run.json' --runs 3 --retries 0 --executable $gui --first-role --analysis 'policy/trajectory_handoff/analyze.py'
--first-role is Boolean, no argument; first official role must be spectator.
For automatic missions substitute run_plan.json. Supervisor owns all native
restarts. Manual mode never automatically restarts the native simulation.
  & $py .\run_simulation.py status
  & $py .\run_simulation.py stop --now
stop --now gracefully stops the owned active attempt/native run. Plain stop
pauses between attempts. GUI alternative: End Simulation in SpaceCRAFT.

LOGS / ANALYSIS
policy/trajectory_handoff/run_logs/<session>/trials/<trial>/<attempt>/
  result.json: task status, public metrics, official_score_captured flag.
  states.txt, commands.txt, radar.txt: shared-runner public observations/actions.
  solver_results.txt: mission jobs, coarse goal, QP and nonlinear near screening.
  monitor_results.txt: local/advisory distinction, sampled candidates, risk status.
  events.txt: actual handoffs, rejected proposals, explicit coast and worker faults.
  policy_telemetry.txt: mode, generations, monitor age and independent worker activity.
  numeric_worker/ and monitor_worker/: one current IPC set and worker.txt per role.
These are runtime outputs, not extra policy revisions or snapshots.
  $s = (& $py .\run_simulation.py status | ConvertFrom-Json).session
  & $py -m policy.trajectory_handoff.analyze --session $s --show
Analysis writes metrics.json, metrics.txt, run_metrics.png under
policy/trajectory_handoff/analysis_results/<session>/. Retain failed attempts.
--replay is a read-only, potentially slow counterfactual using matching model/cache
and public activations. Recorded later states follow ORIGINAL controls. Historical
replay timing cannot reproduce the user's native scheduler or establish completion.
