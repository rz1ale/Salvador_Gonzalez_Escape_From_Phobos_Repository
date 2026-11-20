NAIVE HANDOFF POLICY | v1.0.0 | 2026-11-17

PURPOSE
An adjustable experimental baseline: +Z boost -> attitude setup -> full +X.
No radar processing or online obstacle planner is installed in this policy.
Stage 3 continues while the pointing controller aims +X at the initial closest
point on the 50 m target sphere. This is not a globally optimal mission policy.
Only public own-state/target reads and public acceleration requests are used.
No competition simulation files, native sources, or shared runner are modified.

INSTALL
Extract ZIP CONTENTS into the existing STU4-Solution root. It adds only
policy/naive_handoff and updates AI_readme_simulation_workflow.txt.
Requires the installed shared runner, root nominal_model.json, and the existing
system_identification/model.py backend. No old clone or other policy is needed.
Keep the existing root Controller.py / OriginalController.py.

TERMINAL SETUP (PowerShell)
  $root = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Assets\Repos\STU4-Solution'
  $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
  if (-not (Test-Path -LiteralPath $py)) { throw 'Set $py to your actual STPro bundled Python executable' }
  Set-Location -LiteralPath $root
Bare python may be the Windows Store alias. $py must be set in EACH new terminal.
Use your installed version if the historical v0.55.0 path differs.

PREPARE / SELECT (does not launch the simulator)
  & $py .\policy\naive_handoff\precompute.py
  if ($LASTEXITCODE -ne 0) { throw 'Precomputation failed' }
  & $py -m unittest discover -s policy/naive_handoff/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'Policy tests failed' }
  & $py .\run_simulation.py select --config 'policy/naive_handoff/run_plan.json'
  if ($LASTEXITCODE -ne 0) { throw 'Selection failed' }
  & $py .\run_simulation.py check
Then open Space Teams PRO -> unchanged official COMPETITION_SIM.simconf ->
select spectator. Do not execute Controller.py or entry.py from the terminal.
Each manual GUI opening creates a fresh independent run. No automatic restart.
The shipped precomputed reference is usable with the supplied model; precompute
again after changing settings, mission data, or the model. There is no flight-side
solver or SciPy import. Runtime checks reject stale settings/model assets before
commands; they do not stop the native simulation.

ONE PLACE TO TUNE: settings.json
  label                         distinguish experiments in comparison.txt
  boost_duration_s              default 2; use 1 for the proposed shorter boost
  boost_thrust_fraction         0..1, fraction of maximum pure-axis MODEL delivery
  slew_duration_s / slew_steps   default 9 / 90; duration is never auto-extended
  slew_weights                  offline setup objective (details below)
  tracking_weights              finite-horizon LQR about the setup reference
  forward_thrust_fraction       default 1: maximum pure +X at zero angular request
  hold_weights                  discrete LQR pointing/rate damping in stage 3
  feedback_acceleration_limit_rad_s2   bounds desired angular feedback magnitude
  hold_pointing_deadband_deg / hold_rate_deadband_rad_s   suppress tiny corrections
  control_period_s              default .05; actual SDK read/send times are logged
  test_duration_s               null = keep flying until observed arrival/operator
                                stop; e.g. 30 = finish controlling at 30 s, send zero,
                                and leave native simulation open
  prediction_duration_s         offline nonlinear replay duration, default 30 s
  status_period_s               console update cadence
No setting changes official fuel, score, mass, inertia, thrust, or simulation time.

Example change, while the previous SpaceCRAFT simulation window is closed:
  $p = '.\policy\naive_handoff\settings.json'
  $s = Get-Content -LiteralPath $p -Raw | ConvertFrom-Json
  $s.label = 'boost1_slew5'
  $s.boost_duration_s = 1.0
  $s.slew_duration_s = 5.0
  $s.slew_steps = 50
  $s.slew_weights.fuel = 2.0
  $s | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $p -Encoding UTF8
  & $py .\policy\naive_handoff\precompute.py
  if ($LASTEXITCODE -ne 0) { throw 'Precomputation failed' }
  & $py .\run_simulation.py select --config 'policy/naive_handoff/run_plan.json'
Open the official GUI again. The old run keeps its own settings_used.json.

CONTROL / OPTIMIZATION
The minimum physical rotation that aligns body +X with xf defines one body axis.
For that reduced attitude model, z=[angle; angular_rate], input=achieved alpha:
  z(k+1) = [1 dt;0 1] z(k) + [dt^2/2;dt] alpha(k).
The offline bounded quadratic problem minimizes
  sum dt*(Qangle*(angle-angle_goal)^2 + Qrate*rate^2
           + Reffort*(alpha_positive^2 + alpha_negative^2)
           + Wfuel*fuel_rate/75)
  + Qfinal_angle*(angle_N-angle_goal)^2 + Qfinal_rate*rate_N^2.
Positive/negative delivered authority AND fuel slopes come from the selected
model's embedded native-style allocation, not arbitrary acceleration caps.
This convex problem is solved as bounded linear least squares, with a reported
projected gradient. No Q/R sweep or long nonlinear mission solve is required.
Its optimality applies only to this declared fixed-axis reduced problem.
The finite-horizon LQR tracker uses the same double integrator and independently
adjustable tracking_weights. Its output is feedforward + angle feedback + angular
rate damping; there is no integral term. Stage 3 uses steady discrete LQR gains
for pointing error/rate, with roll angle free and roll rate damped.

Nonlinear replay includes the nominal quaternion/gravity/allocator model and its
separate .025/.020 s channel delays and reporting delays. The replay reports
actual predicted handoff pointing/rate/fuel and integration refinement, separately
from ideal scalar optimizer residuals. It does not establish native fidelity.
At runtime the slew orientation is anchored to measured attitude at the start of
stage 2, and angle is scaled for the live line of sight. Maximum thrust begins at
the configured stage boundary, even if a short setup left residual pointing/rate.
No hidden settle wait, no radar freshness guard, and no policy-issued native stop.

DEFAULT TIMING EVIDENCE
Nominal initial +X misalignment is about 93.55 degrees. For the chosen fixed
axis, the model predicts +.09577 / -.07952 rad/s^2; rest-to-rest minimum is about
8.67 s. Hence the 9 s default. A 5 s experiment remains selectable; inspect its
reported residuals. This is not a general proof for every possible 3-D maneuver.
assets/precompute_report.txt and prediction.png contain current model predictions.

PUBLIC REQUESTS / MAXIMUM THRUST
The current allocator maps requests to delivered acceleration with a mean-period
normalization. The pure +X saturation knee is about .0036658 m/s^2 PUBLIC request;
its MODEL delivered acceleration is about .1712 m/s^2. Sending 10 instead does not
increase maximum predicted pure-axis delivery. The analogous +Z knee is computed.
The policy constructs translation/angular request RATIOS at these knees. For
full-thrust phases it multiplies the entire six-vector by ONE common factor until
one native norm cap is reached. Pure boost/forward therefore sends .5 m/s^2; a
mixed command can reach the .05 rad/s^2 angular cap first. This preserves relative
angular authority, retains saturated mean delivery, and avoids depending on the
fitted .021 s period to guarantee saturation. It never clips/scales the groups
independently to create extra authority. Partial-thrust fractions retain their
mean-model amplitudes. Shared allocation means angular corrections can reduce
forward delivery; full X does not promise full delivered X plus free rotation.
Only current nominal-model values are used; the physical native response is tested
by the operator. There is no flight-side LP or assumption that raw u=achieved a.

FUEL / SCORING (accounting.py, NOT settings.json)
Official supplied constants: Isp=80 s, g0=9.8 m/s^2. Given nonnegative native LP
allocation lambda, L=max(lambda): on_time_i=lambda_i*min(actual_native_dt,L)/L;
fuel=sum(jet_force_i*on_time_i)/(80*9.8). The zero case returns zero.
MODEL estimates use the fitted .021 s mean period; this is not a known native
timestep. The exact native period and allocator firing record are not accessed.
Actual public Stats/FuelUsed is PRIMARY for run comparison. Do not substitute a
quadratic effort penalty or model estimate for measured competition fuel.
R_effort is a tunable regularizer; W_fuel scales a fuel cost without altering the
fuel law. Fixed-axis fuel slopes are exact for that reduced mean allocation ray;
mixed commands are evaluated through the full model only in offline replay.
Winning score estimate only AFTER observed arrival:
  S=50+100*(120/t - 2*damage_J/209200 - max(fuel_kg/75,0)).
The native scorer's own clock/polling/result is authoritative. Never report a
winning score for a partial run or call this baseline a maximum-score solution.

STOP / COMPLETE
Normal: on first public read within 50 m, write result, send zero, remain connected.
Configured test duration: same keep-open behavior, outcome not_assessed.
Operator: close/end the simulation in the SpaceCRAFT GUI. Terminal alternative:
  & $py .\run_simulation.py stop --now
The runner's explicit operator-stop path requests native shutdown. Zero thrust
does not brake existing velocity. There is no default mission timeout here.
10054 alone does not establish a crash; inspect result.json / summary / native logs.

ANALYSIS AFTER EACH RUN (PowerShell)
  $session = (& $py .\run_simulation.py status | ConvertFrom-Json).session
  & $py .\policy\naive_handoff\analyze.py --session "$session" --show
  & $py .\policy\naive_handoff\analyze.py --all --show
The first writes per-attempt metrics.txt/run_metrics.png. --all rebuilds the one
canonical comparison.txt (tab-delimited), comparison_details.json, comparison.png.
Logs: policy/naive_handoff/run_logs/<session>/trials/<trial>/<attempt>/
Analysis: policy/naive_handoff/analysis_results/
Telemetry: naive_handoff_telemetry.txt is JSON-lines .txt, beside the runner's
states.txt/commands.txt/events.txt. Raw public stats, six requests, body angular
rate, pointing error, read/send times, and measured stage transitions are logged.
Analysis supports partial/interrupted logs, includes the final arrival read, and
preserves the first observed arrival. Per-stage fuel uses observed boundary reads;
stage timing resolution is explicitly noted. It reports startup time, gaps, damage,
handoff pointing/rate, fuel by stage, final distance, and progress per fuel.
Partial tests should be compared at equal duration. They have no winning score.
Optional slower command-history MODEL fuel estimate:
  & $py .\policy\naive_handoff\analyze.py --session "$session" --model-fuel
Requires the model hash from that run; public stats can always be analyzed alone.

FILES
settings.json         policy timings/weights, the main tuning surface
mission.json          supplied own initial state/public target for offline work
precompute.py         constrained attitude setup, gains, nonlinear replay, plots
control.py            lightweight flight math + scheduled feedforward/feedback
entry.py              run(context), public SDK I/O via existing runner
accounting.py         official fuel/scoring formulas; no tunable constants
analyze.py            per-run and comparison accounting/plots
run_plan.json         existing runner selection; manual, independent GUI sessions
assets/               generated reference, manifest, report, prediction
tests/test_policy.py  meaningful offline physics/accounting/runner-loader checks

DEFAULT VERIFICATION
See VALIDATION.txt for measured offline checks and limitations. Native execution
on the user's installed Space Teams PRO remains the next test.
SciPy tools: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html
