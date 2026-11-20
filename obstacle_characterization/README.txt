OBSTACLE_CHARACTERIZATION: tracker v0.1.1 + ellipsoid queries -- 2026-11-14
SHARED RADAR-ONLY SERVICE FOR ALL POLICIES

PURPOSE / BOUNDARY
Import this package from any existing run(context) policy. It converts permitted
public radar + own-state history into one versioned, JSON-compatible obstacle
frame. Tracking/geometry/prediction/collision queries are shared. Policy planning,
objective weights, vehicle commands, mission completion and restarts remain in
the policy/shared runner. No camera, SDK connection, native configuration change,
private obstacle truth, allocator telemetry, hidden seed use, or -old dependency.
Read ROOT/AI_readme_competition_overview.txt first.

INSTALL
Extract ZIP CONTENTS into the current STU4-Solution root. Adds only this folder.
No Controller.py, active_run.json, native .simconf, model, runner or system-ID
flight files are overwritten. NumPy/SciPy use the existing STPro interpreter.

FILES
  __init__.py: public API exports.
  radar.py: own-pose history, public clock bounds, filtering and association.
  collision.py: forecast tubes, threat ranking, swept trajectory checks.
  runtime.py: RadarStream adapter for the existing context interface.
  settings.json: sole shared radar/tracker default settings.
  replay.py: local offline replay/diagnostics from existing public recordings.
  smoke.py / smoke_run.json: optional read-only native interface check + RUNNER
    plan, not another simulator configuration.
  tests/test_radar.py: synthetic functional/regression tests, no native launch.

POLICY INTEGRATION (SAME ENTRY FOR MANUAL/AUTOMATIC)
  from obstacle_characterization import RadarStream, predict_objects
  from obstacle_characterization import check_trajectory, rank_threats

  def run(context):
      obstacles = RadarStream(context)
      while ...:                         # existing policy loop
          own = context.state()
          frame = obstacles.poll(own_state=own)
          # frame -> controller/planner; controller chooses/sends commands

poll() caches the latest frame between polls (default 0.2 s) but keeps supplied
own-state history current. Reused frames retain their original generated_sim_s;
callers must not pretend they are fresh. Passing own avoids a duplicate state RPC.
All SDK reads stay on the caller/controller thread. The adapter uses only
context.state(), radar(500) and clock(); it sends no commands or lifecycle actions.
Core RadarTracker is SDK-independent for offline/worker use:
  tracker = RadarTracker(settings_dict)
  tracker.add_own_state(public_state_packet)
  frame = tracker.update(public_radar_rows, receive_sim_s)
No background threads/connections are created. Synchronize ownership if using a
worker; do not call SDK objects concurrently without verified SDK support.

RUNNER DEPENDENCIES
Add these solution-relative paths to EACH policy plan's dependencies list:
  obstacle_characterization/__init__.py
  obstacle_characterization/radar.py
  obstacle_characterization/collision.py
  obstacle_characterization/runtime.py
  obstacle_characterization/settings.json
The runner then detects shared-module/settings edits on resume. Each live adapter
also logs source/settings hashes. Change sources/settings between runs, not midrun.

COMMON FRAME CONTRACT: schema_version=1; frame=Phobos_body_fixed; units=SI
Top-level:
  producer, producer_version, sequence, generated_sim_s, state_order
  motion_model, acceleration_spectral_density_m2_s3, default_query
  clock: nominal_offset_s, offset_bounds_s, nominal_to_bound_s, mode, exact
  own_state: radar_origin_P_m, velocity_P_m_s, velocity_source, read/extrapolation
  objects: every assessed track AND unresolved association alternative
  coverage: scope, guarantees_free_space=false, object counts, conditions
  diagnostics: fresh/duplicate/invalid/ambiguous/matched/birth/expiry counts,
    mean position innovation and processing_wall_ms

Every object:
  id: local estimated identity, NOT a native debris ID; unique within one tracker
  kind: track | unassociated_observation
  sources: ["radar"]                    # extensible for future sensor fusion
  status: tentative | confirmed | coasting | stale | unresolved | lost | lost_unresolved
  state_epoch_sim_s: same as frame.generated_sim_s
  center_P_m[3], velocity_P_m_s[3]
  covariance_P[6][6]: [x,y,z,vx,vy,vz], SI units, already predicted to frame epoch
  radius_m, radius_sigma_m, extra_position_margin_m
  extent_kind=radar_sphere_proxy; bounding_geometry_verified=false
  last_measurement_sim_s, measurement_time_bounds_sim_s
  measurement_age_s, measurement_age_bounds_s, last_received_sim_s
  independent_observations, association_ambiguous, last_radial_relative_m_s

Use generated_sim_s to propagate to FUTURE ABSOLUTE simulation times. Do not add
measurement age again: current object states/covariances already include it.
last_radial_relative_m_s is an old raw scalar, not current full obstacle velocity.
Never silently discard tentative/unresolved objects; they are possible hazards.
Clients should tolerate added optional fields. Changing units/frame/field meaning
requires a new schema version. Camera processing can later update the same track
state/contract rather than altering every policy.

WHAT IS IMPLEMENTED / WHAT IS ASSUMED
1. Z_i = [relative_center_P(3); closest_proxy, radial_rate, diameter;
          sensor_time, 0, 0]. Only the documented public rows are consumed.
2. Each positive-timestamp measurement is deduplicated by full payload. Reordered
   rows are not identities. Repeated reads do not shrink filter covariance or
   count as new evidence. Changed same-epoch payloads are retained as alternatives.
3. Own positions are interpolated at measurement time, not simply added to old
   radar coordinates at receipt time. Position-history derivatives estimate the
   relevant origin velocity, avoiding an assumption that SDK velocity is the
   derivative of the reported point. Very early fallback velocity is uncertain.
4. radar_origin_offset_B_m is FROM SDK reported position TO radar-relative origin;
   default zero is a declared origin assumption. It is NOT the nominal model's
   fitted reporting-to-effective-COM offset. Do not apply that offset twice.
5. Native supplied Radar.py timestamps are relative to its runtime origin.
   Default auto mode assumes that origin occurs on/after the sim epoch. It uses
   offset 0 nominally and bounds true offset by:
       [0, minimum_over_packets(receive_time - latest_positive_sensor_time)].
   That is a bound, not exact synchronization. No guessed network latency is
   subtracted. Common clock uncertainty is retained OUTSIDE filter covariance
   as extra_position_margin_m, using the declared obstacle-speed assumption plus
   observed own speeds. It is never averaged away as independent sensor noise.
   Set timestamp_offset_s + timestamp_offset_uncertainty_s explicitly only when
   established. Zero initialization stamps are not treated as measured epochs.
   Missing own history/future/invalid/stale packets remain explicit conditions.
6. Six-state constant-velocity filters use continuous white-acceleration process
   covariance. Position and radial-velocity updates use Joseph covariance form.
   Initial unknown absolute transverse velocity: mean zero, std 2 m/s; this
   does NOT mean transverse motion is known zero. These priors require tuning
   from permitted records; no per-debris physics/ephemeris is read.
7. Global one-to-one gated assignment uses position, radial velocity and size.
   Ambiguous observations remain alternative hazards; tracks are not forcibly
   merged. Confirmation requires 3 independent epochs spanning >=1 s.
   Covariance is conditional on the assumed associations/models; it is not a
   calibrated mission collision probability.
8. Diameter/closest-point proxy produce a noisy spherical extent estimate.
   Physical mesh containment is unverified; shape_padding_m remains explicit.
   Deletion/expiry, missing observations and limited nearest-list coverage do
   not prove free space. Capacity loss stays flagged through a retention window.

ELLIPSOID QUERY EXTENSION (2026-11-13)
Tracking/noise settings, schema-v1 stream and existing sphere-query defaults are
preserved. This extension originally passed 30 offline checks; v0.1.1 now passes
40 including lifecycle and birth-initialization checks. Native enclosure/probabilities remain
unqualified. This is a solution-level numerical update, not a native sensor edit.

Optional policy query:
  risk = check_trajectory(frame, times, centers_P_m, vehicle_radius_m=4.,
      clearance_m=3., uncertainty_geometry='ellipsoid')
For a fixed UNIT normal n, an ellipsoid's support radius is
  k*sqrt(n.T @ position_covariance @ n).
The query tries the relative-center normal and a closest-ellipsoid-point normal.
For every WHOLE segment it checks the minimum endpoint center projection against
extent + user/vehicle/clock/shape allowances + maximum projected uncertainty.
For CV+white-acceleration covariance, projected variance is a convex cubic in
nonnegative forecast time, so that maximum occurs at an endpoint. A positive
margin separates the declared moving ellipsoid+sphere for the entire chord.
A nonpositive margin is potential conflict, not proof of physical collision.
The same covariance and k are used; no tangential uncertainty is erased.

position_support_sigma(covariance, unit_normal) and
ellipsoid_separating_normal(relative_center, covariance, multiplier=3.) are
available for optimization. predict_objects adds extent_radius_with_margins_m
(object size + size/shape/clock allowances, excluding POSITION uncertainty,
vehicle and clearance). Its existing inflated_obstacle_radius_m remains the
legacy largest-axis sphere. Use the covariance plus directional support when
constructing ellipsoid half-space constraints; do not add the sphere again.
Query results label uncertainty_geometry and separating_projection_m/normal_P.
The projection margin may differ from Euclidean center distance. All retained
objects, unresolved hypotheses and coverage conditions remain included.

RETAINED OCCUPANCY EXTENSION (2026-11-14)
Optional setting track_expiry_mode=retain_occupancy separates association expiry
from hazard lifetime. The default drop_unassessed preserves legacy clients.
Tracks older than track_timeout_s (default 12 s) leave active association but
remain in every forecast/query as
status=lost, occupancy_retained=true, association_active=false. Expired unresolved
alternatives remain as lost_unresolved. Their original posterior/measurement
epochs persist; full elapsed CV+white-acceleration covariance is propagated to
the current frame and then through the requested future horizon. Age is not
reset, covariance is not shrunk, and reacquisition is not assumed.

coverage/diagnostics report retained_expired_tracks and
retained_expired_unresolved_observations. expired_tracks remains an event count;
it does not alone make a fully represented retained frame unassessed. Missing/
invalid measurements, bad clocks, missing own history and true capacity losses
remain conditions. No absence of measurement implies free space.

Active+retired tracks share max_tracks; current+retired unresolved observations
share max_unresolved_observations. If occupancy must be dropped at capacity,
retained_occupancy_capacity_loss stays flagged for the rest of that run. Hypotheses
may overlap or duplicate real objects: retaining them is conservative, not proof
of their physical identity. No secret ID, deletion oracle or simulator edit is
used. Long-unobserved occupancy can legitimately make planning infeasible.
The new seven lifecycle tests include covariance/identity retention, blocking
intersections, unresolved alternatives, invalid data, capacities and mode validation.

TWO-POSITION BIRTH INITIALIZATION (2026-11-14)
Optional birth_initialization=two_position; default radial_prior preserves old
clients. For a ONE-hit track with dt>=confirmation_span_s, a rejected joint
radial-rate gate may fall back to the existing position+size gate. Global
assignment and ambiguity checks still apply. A unique accepted pair initializes
  v=(p_new-p_old)/dt; P_pp=R_new; P_pv=R_new/dt;
  P_vv=(R_old+R_new)/dt^2 + q*dt/3*I.
The latest radial rate then updates the full posterior. The old one-hit velocity
prior is superseded; mature tracks, same-epoch alternatives and ambiguous matches
are not forcibly reinitialized. This is an explicit estimator/model choice,
not an assertion that every geometric match is physically correct. Noise scales
and geometric/statistical query margins are unchanged.
Objects label velocity_initialization. Diagnostics report per-packet and total
birth_reinitializations. Three tests cover a radial-rejected unique birth with
cross covariance, an ambiguous pair and a mature-track rejection.

PREDICTION / POLICY-FACING QUERIES
  forecast = predict_objects(frame, future_times_sim_s)
  # For solver use without nested Python-list conversion:
  forecast = predict_objects(frame, future_times_sim_s, array_output=True)
  # Per-object centers, position covariances and inflated obstacle radii.
  # Vehicle radius and deliberate user clearance are added separately.
Default forecast output is JSON-compatible. array_output=True keeps NumPy arrays
and marks numeric_representation=numpy; do not JSON-serialize that optional form.

  risk = check_trajectory(frame, future_times_sim_s, vehicle_centers_P_m,
      vehicle_radius_m=policy_settings["vehicle_radius_m"],
      clearance_m=policy_settings["clearance_m"],
      vehicle_acceleration_bound_m_s2=policy_settings["position_acceleration_bound"])

Vehicle radius must enclose the actual shuttle about the supplied trajectory
center, at every attitude. Centers can use the SDK origin or model effective
point only with the corresponding enclosing radius. This module invents neither
vehicle dimensions nor a policy-specific safe path.
Checks cover closest RELATIVE approach continuously inside every linear segment,
including obstacles crossing between otherwise clear endpoints. Optional a_max
adds a_max*segment_dt^2/8 to cover the vehicle's deviation from its chord.
Obstacle means are CV; covariance/clock/size/shape allowances inflate them.
The acceleration/chord allowance requires an actual acceleration bound. Native
impulses, vehicle-model error and additional command latency need policy-level
allowances; the segment test is not a native continuous-motion safety proof.
Default inflation multiplier=3 and shape padding=1 m are assumptions, NOT safety
certification or calibrated 99.7% mission coverage. User clearance is independent.
Required separation = vehicle radius + object radius + user clearance + size
allowance + shape allowance + common-clock allowance + forecast uncertainty.

Risk assessment: potential_collision | no_predicted_intersection | unassessed.
certified_safe is ALWAYS false. The first is a conservative model conflict, not
proof of actual contact; the second concerns only represented objects/assumptions.
Queries beyond forecast_horizon_s (default 10 s) are unassessed; forecasts reject
unsupported times. Extend the model horizon deliberately after evaluating error.
rank_threats(frame, vehicle_radius_m=..., clearance_m=..., horizon_s=...) screens
constant-own-velocity motion and returns all objects ordered by inflated clearance.
It does not generate a maneuver. Policies may reuse forecasts within an optimizer;
do not repeat tracker updates or Python association inside cost/Jacobian evaluations.

LOGGING / LOCAL ANALYSIS
Native adapter: lightweight obstacle_diagnostics.txt JSON Lines in the existing
attempt output directory, at 1 Hz by default. Existing runner raw states.txt and
radar.txt already provide replay inputs. Optional RadarStream(...,
log_full_frames=True) additionally writes full obstacle_stream.txt; can be large.
No duplicate raw-data trees or controller snapshots are created.

PowerShell at solution root:
  $py = Join-Path $env:LOCALAPPDATA 'SimDynamX\STPro\Versions\v0.55.0\py\Scripts\python.exe'
  & $py -m unittest discover -s .\obstacle_characterization\tests -v
Replay the already completed scout locally, without starting a simulator:
  & $py -m obstacle_characterization.replay --session 'system_identification/data_sets/20261106_054253_3143e1a4' --show
Or use --attempt 'solution/relative/attempt_folder' instead of --session.
Outputs: obstacle_characterization/analysis_results/SOURCE/replay_TIME/
  summary.txt, metrics.json. --frames explicitly adds full standardized .txt
  streams. Inputs remain unchanged; files are hashed; processing is causal.
Replay timings exclude SDK calls and frame-file serialization. They do not
establish real-time performance or obstacle-truth accuracy. No private truth used.

OPTIONAL NATIVE INTERFACE SMOKE (USES EXISTING RUNNER / OFFICIAL SIM)
  & $py .\run_simulation.py select --config 'obstacle_characterization/smoke_run.json'
  & $py .\run_simulation.py check
Then open unchanged STU4-Sim/Sims/COMPETITION_SIM.simconf and select spectator.
The 4-second task sends NO vehicle commands. It checks finite standardized frames
and at least one assessed object, commits a result and requests normal native
STOP. No automatic restart in manual mode. A receiver-closed/10054 notice after
that intentional stop alone does not imply a crash. Read the result/status.
Automatic option, after closing SpaceCRAFT completely and defining $gui:
  & $py .\run_simulation.py auto --config 'obstacle_characterization/smoke_run.json' --runs 1 --retries 0 --executable $gui --first-role
--first-role is boolean and requires spectator to be the first native role.
This smoke establishes an interface path, not detection accuracy or collision
safety. It does not change system-ID qualification receipts. Restore/select the
desired policy afterward; the smoke runner plan is an explicit selection.

VERIFICATION / LIMITS
Functional tests cover moving ego origin, six-state prediction, noisy shuffled
targets, nonidentity radial velocity, deduplication, same-epoch alternatives,
clock bounds, missing history, ambiguous association, capacity/expiry, schema
errors, swept collisions, uncertainty/margin growth, interpolation allowance,
runtime adapter, offline replay and the observation-only smoke task.
Supplied real public attempt e9920076 replays 342 radar packets x 200 rows.
This checks parsing/tracking execution, not native obstacle-truth accuracy.
Native live smoke has NOT been executed in the authoring environment.
No original runner/common/runtime/flight/settings/native file was changed.

AUTHOR CHECKS (DISTINGUISH OFFLINE FROM NATIVE)
22 functional/regression tests passed. Existing runner accepts smoke_run.json.
Isolated-package CLI replay of the supplied public attempt passed.
200-row real-record replay (342 frames): about 12 ms median / 17 ms p95 processing
here, excluding native I/O and optional frame-file serialization.
Synthetic 500-row asynchronous replay: about 11 ms median / 25 ms p95 here.
Separate 500-object / 21-knot query: about 6 ms median; NumPy-array forecast about
4 ms median. Full JSON-list forecast conversion can cost substantially more.
These author-host timings (single BLAS thread) are examples, not Windows deadline
guarantees. Observer/solver integration and live native latency remain to measure.
Association identities are estimates; replay completion is not identification
accuracy, and object count can include multiple hypotheses for the same object.
