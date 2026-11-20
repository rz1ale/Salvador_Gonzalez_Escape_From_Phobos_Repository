SYSTEM IDENTIFICATION CAMPAIGN v1.0.0 -- 2026-11-05
Entry: system_identification/campaigns/identification.py -> run(context)
Requires shared runner v1.3.0 supplied in this ZIP. Read the competition overview
and AI_readme_simulation_workflow.txt. Only unchanged official competition sim.

INSTALL / FIRST NATIVE SMOKE
1. Close SpaceCRAFT and run_Server. Extract ZIP CONTENTS into STU4-Solution.
   This upgrades shared runner files and adds system_identification. It contains
   NO Controller.py, OriginalController.py, active_run.json, .simconf or fitted model.
   The portable model backend and fitting code are included.
   Your installed controller dispatch remains; the official bootstrap is untouched.
2. PowerShell in STU4-Solution:
     $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
     if (-not (Test-Path -LiteralPath $py)) { throw 'Use the installed STPro Python path; see workflow README' }
     & $py .\run_simulation.py selftest
     if ($LASTEXITCODE -ne 0) { throw 'Offline tests failed' }
     & $py .\run_simulation.py select --config 'system_identification/campaigns/smoke_tests.json' --case SMOKE01_LIFT --runs 1
     & $py .\run_simulation.py check
3. Open the unchanged Repos/STU4-Sim/Sims/COMPETITION_SIM.simconf in STPro;
   select spectator. This smoke commands takeoff to 25 m launch-relative height,
   settles and holds, then requests native STOP. It does NOT restart or land.
4. Report the status and preserve any failed attempt:
     & $py .\run_simulation.py status
   Expected after native verification: complete, attempts=1, task_failures=0.
   Takeoff typically needs roughly a minute in the offline fixture; the 2-second
   zero-thrust infrastructure smoke was NOT an actuator/takeoff verification.

CURRENT NATIVE EVIDENCE (UTC IDs; evidence types distinguished below)
SMOKE01_LIFT passed in 20261106_014619_21fa512d and 20261106_014811_e599bd82:
84.998/84.305 s flight duration, 0 J damage. Earlier 20261106_014353_58408978
failed no_upward_motion; cause unresolved. All three ended by Controller-requested
native stop. Receiver-closed errors followed that stop during native cleanup.
Subsequently, user reported the manual all-three smoke gate passed. Automatic
session 20261106_052840_e65667dc completed all three with exact sysid_*_complete
reasons, attempts=3, trials_finalized=3, task_failures=0 (terminal evidence only).
The normal automatic ID launch/stop/relaunch sequence is verified. That session's
raw logs/launch.json were not uploaded; status alone does not verify role order.
Next: scout.json (28 flights). Existing code/settings remain unchanged. Preserve
failed attempts and referenced smoke logs; temporary collectors are disposable.

THREE SMOKES / NATIVE GATE
SMOKE01_LIFT: nonzero public thrust, upward motion, bounded climb, attitude/position
  stabilization and hold. No-upward-motion check is 3 simulation seconds after
  first own-state capture; no fixed artificial pre-thrust delay is added.
SMOKE02_PULSES: takeoff; local net-gravity coast; recover; isolated +X linear pulse
  (.003 raw, .8 s); recover; isolated +X angular pulse (.003 raw, .15 s); coast and
  recover. Signed motion/rate responses must be positive and measurable.
SMOKE03_COAST_MIXED: takeoff; net-environment coast; four short six-input segments;
  coast and recover. Checks joint-input logging, bounded motion and recovery.
No intentional collision is included. Damage/proximity/latency/envelope violations
end the attempt as FAILURE with retained raw logs and a reason; they do not count
as successful probes. No task-failure retries are enabled by default.

After the manual smoke, close SpaceCRAFT completely. Execute all three via the
SAME entry and root controller:
  $gui = Join-Path $env:PUBLIC 'SimDynamX\STPRO\Apps\Main_SG\SpaceCRAFT\SpaceCRAFT.exe'
  if (-not (Test-Path -LiteralPath $gui)) { throw 'Use the installed SpaceCRAFT.exe path' }
  & $py .\run_simulation.py auto --config 'system_identification/campaigns/smoke_tests.json' --executable $gui
Choose spectator at each role prompt, or append --first-role once the first role
is confirmed spectator. It is a boolean switch with NO role-name argument:
  & $py .\run_simulation.py auto --config 'system_identification/campaigns/smoke_tests.json' --retries 0 --executable $gui --first-role
The flag passes native -firstRole. No role-picker/configuration edits are needed.
Status alone cannot establish unattended role choice; verify it visually once.
All three should finish: status complete, trials_finalized=3, task_failures=0.

Science runs require all three NEW smoke receipts in smoke_status.json. Gate
checks the referenced actual result and supervisor result; a pending/failed/latest
attempt cannot reuse an older pass. Matching code/settings, official configuration,
Systems Python, matching jet/thrust/RCS CSV source fingerprints and NumPy version
are required. Files are fingerprinted read-only; no private debris/ephemeris table
or runtime allocator/tick telemetry is read. This is a project workflow guard,
not an organizer authenticity/security certification. Recheck after native binary
or SDK updates too; fingerprints do not capture every platform dependency.

UNATTENDED RETRIES (OPTIONAL v1.2 SCHEDULER; NO FLIGHT/SAFETY CHANGES)
Use --retry-schedule 3 5 on auto for 3 total main-pass attempts, then failed-only
catch-up up to 5 total attempts each. Success stops retries for that trial. All
failed data remain recorded; maxed-out trials appear in status.unresolved_trials.
The switch is a launcher policy frozen in plan.retry_schedule, not a flight
parameter. Do not combine it with --retries. Smoke/default JSON definitions retain
zero retries/pause for validation; science commands below explicitly opt in.
Operator stop/controller errors/missing smoke qualification still pause. For an
ongoing paused campaign, update the same session via set-retries; complete steps,
source-hash safeguards and resume syntax are in workflow section 9A. No install,
new controller copy or native-smoke repeat is required for this scheduler update.

LOCAL ANALYSIS (NORMAL PATH; NO UPLOAD REQUIRED)
  & $py .\system_identification\analysis_code\summarize.py --latest --show
--latest uses the saved active ID session pointer, with no directory guessing.
Read report.txt/assessment.json for verdict, measured responses, coverage, timing,
coupling, failures and the next command; original metrics/audit files are retained.
Complete clean intervals from failed flights are listed separately, not counted
as successful whole trials. No numeric holdout inspection unless explicitly opted in.
Append --analysis 'system_identification/analysis_code/summarize.py' to auto/resume
for offline assessment after completed collection and native cleanup. Source is
pinned in plan.post_analysis. No solver/report work runs in the controller loop.
summarize.py does acquisition QA only. Actual fitting/export now uses:
  & $py .\system_identification\analysis_code\fit_model.py --latest --show
For the completed scout, use --session 'system_identification/data_sets/20261106_054253_3143e1a4'
instead of --latest if another session is selected. No new flights required.
It writes model.json, metrics.txt/metrics.json, fit_diagnostics.json, report.txt and
per-flight truth/prediction + error PNGs in analysis_results/SESSION_ID/model_TIME.
Training-only scout outputs are explicitly not independent validation. Declared
validation is excluded from fitting; testing remains sealed by default. Requires
NumPy/SciPy/Matplotlib. Add --publish to explicitly select nominal_model.json;
otherwise no policy-facing model is changed.
The fitter reads the active official JetSelectionActuator.JetConfigName to locate
the table (currently STU4-Sim/Data/SpaceBuggyJetConfigNoMains.csv). It no longer
assumes the exported filename jet_config.csv. No native files are edited. Full model assumptions/API: reference
section 16 and workflow 9C. It is a local structural candidate, not qualification.
Workflow section 9B gives the full local contract. Routine AI inspection is optional.

SCIENCE PLANS (FINITE; ONE FRESH NATIVE FLIGHT PER CASE)
campaigns/saturation.json: 48 cases. 6 body axes x 2 signs x 4 raw amplitudes.
  Linear: .0002, .001, .004, .5 m/s^2 requests; 1 s nominal exposure.
  Angular: .0001, .0005, .003, .05 rad/s^2 requests; .2 s nominal exposure.
  Other five requests are exactly zero during isolated pulses. Braking, coasts
  and stabilization are separate labeled phases. Stopping a pulse early censors
  it; never claim its planned saturation level was tested.
campaigns/coasts.json: 4 cases. Heights 22/28 m and two modest attitude offsets.
campaigns/timing.json: 8 cases. Width changes, reversals and low/high requests.
campaigns/coupled.json: 4 cases. Joint translation/rotation and shared saturation.
campaigns/validation.json: 8 complete held-out profiles: 4 validation, 4 testing.
campaigns/full_campaign.json: 72 science cases (64 training + 4 validation + 4 test).
campaigns/scout.json: 28 training cases: low/high signed axes plus coasts/coupled.
Scout does not establish a saturation plateau from two points alone; run the
intermediate levels/repetitions when the practical smoke/scout results justify it.
All JSON plans are definitions; they share ONE experiment implementation.

Start with scout after smokes, not an unobserved long batch:
  & $py .\run_simulation.py auto --config 'system_identification/campaigns/scout.json' --retry-schedule 3 5 --executable $gui --first-role --analysis 'system_identification/analysis_code/summarize.py'
Full plan when ready:
  & $py .\run_simulation.py auto --config 'system_identification/campaigns/full_campaign.json' --retry-schedule 3 5 --executable $gui --first-role --analysis 'system_identification/analysis_code/summarize.py'
Single manual signed-axis case:
  & $py .\run_simulation.py select --config 'system_identification/campaigns/saturation.json' --case SAT_TX_P_A01 --runs 1
Then open the official sim yourself. A manual SET uses:
  & $py .\run_simulation.py manual --new --config 'system_identification/campaigns/saturation.json'
Open a new official GUI simulation once per pending trial. ID tasks explicitly
request native stop after result commit in both modes; only the supervisor
relaunches. Defaults in OriginalController/ordinary tasks still leave manual
simulations open unless they explicitly request stop.

GUARDS / CONTROL ASSUMPTIONS
campaigns/settings.json is the single source of numeric acquisition settings.
Height is (reported p - first reported p) dot fixed radial-up at launch; it is
NOT actual terrain clearance. Target 25 m. Science band 20..30 m; hard ceiling
35 m. Launch transit begins at 0 m with a -0.5 m floor. The 12 m airborne floor
arms only after crossing 13 m; after first settling, the science band is enforced.
No terrain query/private geometry is used. These are configured conservative
defaults retaining the historical ceiling, not proof that a route is unobstructed.
Speed <=.9 m/s; launch-plane lateral displacement <=15 m; body rate <=.12 rad/s;
attitude excursion <=.45 rad from launch. Delay/braking guards also bound predicted
vertical, lateral and angular stopping excursions. Default delay .5 s; net braking
.025 m/s^2 and .01 rad/s^2 are assumptions, to be checked with the new smokes.
No larger envelope is introduced automatically. Modify/review settings once and
rerun all smokes when changes are needed; never weaken native physics/configuration.

The staging/recovery controller uses reported truth directly, passive P-to-B xyzw
rotations, source-informed h=.02072028 s request scaling, and a local net-gravity
estimate from clean coasts. h is a historical prior, not live native tick access
or the controller period. Source/actuation differences can therefore fail a smoke;
fix the solution mapping using evidence, never the native actuator or configuration.
Requested control updates are 20 Hz; actual read/send brackets and gaps are logged.

Radar guard uses up to 500 public observed spheres, radial rates, measurement age,
known own transverse motion and bounded unknown transverse debris motion. Range/
position uncertainty and own acceleration are inflated. Coverage is incomplete;
no detections do not prove a clear path. This is a guard, not a complete obstacle
avoidance planner. Contact-free inside-field data may be useful; damage, truncated
commands, stale/gapped samples and guard interventions are explicitly recorded.
Positive initial damage rejects the nominal flight. Nominal clean intervals stop
before detected contact; raw tails are retained. No automatic post-impact reentry.

DATA / AUDIT / MODEL BOUNDARY
Session: system_identification/data_sets/UTC_TIMESTAMP_UUID/
  plan.json, ledger.json, trials/TRIAL_ID/attempt_N_UUID/
Attempt: shared states/commands/radar/events .txt logs, metadata.json, quality.json,
  result.json and automatic supervisor_result.json; telemetry.txt is a flat TAB-
  delimited own-state/request/phase table. Read times precede publication of the
  row's command; use send-time history, not a row-aligned acceleration assumption.
Metadata includes own mass/inertia, frames, launch reference and priors. quality
records complete clean phases, contact cutoff and censored/interrupted phases.
Published request exposure has time brackets; actual native impulse timing is
unobserved. Pulse diagnostics use the following coast to allow delayed reports to
include the whole pulse, subtract local gravity, and divide by request exposure.
They are motion-inferred interval estimates, not private allocator observations.

Audit one explicit session, without launching a simulation:
  & $py .\system_identification\analysis_code\summarize.py --session 'system_identification/data_sets/SESSION_ID'
Outputs: analysis_results/SESSION_ID/audit_TIMESTAMP/metrics.txt, audit.json,
  authority_observations.json. Add --plots smoke or --plots all for plots using
Matplotlib in the analysis environment; default audit needs no Matplotlib.
Held-out numeric trajectories remain unopened by default;
--include-heldout explicitly unseals them. Training-only plateau candidates need
repeat evidence; the script does not declare an entire six-axis reachable envelope.
An audit produces no fitted nominal_model.json, uncertainty covariance or official
score. The portable nonlinear backend/model fit is analysis_code/fit_model.py; see above.
There is no automatic model promotion/overwrite.

SOURCE / MAINTENANCE
Entry: campaigns/identification.py. Pure math: campaigns/flight_math.py.
Plan builder: campaigns/make_plans.py (writes solution JSON only; no simulator).
Rebuild plans only when intentionally changing experiment definitions:
  & $py .\system_identification\campaigns\make_plans.py
Use git/source versions rather than cloning campaign controller/launcher files.
Each new session snapshots code/config/model/dependency hashes. Source changes
prevent resuming old sessions. Preserve previous attempts and start a new session.
Pause/status/resume use run_simulation.py; see the main workflow README. Stop a
long automatic batch with 'stop' (after current attempt) or 'stop --now' (current).
If a smoke fails, share its explicit session ZIP, status and quality/result reason.
Do not immediately rerun the full campaign or remove its failed attempt.

VERIFICATION STATUS
Shared runner v1.0 normal path was natively verified by the user: manual 1/1 and
automatic 2/2, interactive spectator. The normal native ID smoke suite and scout completed as recorded above. New
model fitting/replay is software-tested on synthetic data; actual native prediction
quality must come from the local saved-data fit metrics. The offline fixture includes
shared input limits, 6DOF kinematics, delayed reports and a reporting-point offset.
It remains a mathematical fixture, not the actual engine or a competition score.
