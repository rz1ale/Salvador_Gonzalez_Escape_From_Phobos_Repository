V4 — CONFIGURABLE INITIAL REORIENTATION / BODY-DIRECTION THRUST
================================================================

EXPERIMENTAL: OFFLINE MODEL VALIDATION IS NOT NATIVE RELIABILITY EVIDENCE.
The fitted model missed the terminal native position of a recorded V2 U2
flight by ~57 m in command-timeline free-run, despite ~0.005 kg fuel error.
Do not treat predicted score or capture as guaranteed.

INSTALL
-------
Extract the archive directly into STU4-Solution preserving:
  policy/bangbang_coast_v4/

The archive contains ONLY this new directory. It has no V1/V2/V3
controller files, no protected Controller.py, no active_run.json, no
nominal_model.json, no simulator files, no sim_core files. V2/V3 remain
unchanged and selectable.

REQUIRES existing root files:
  nominal_model.json
  system_identification/model.py
  sim_core/... and run_simulation.py (existing runner)

The existing model, public context interface, and conservative V3
coast/recovery implementation are reused as V4-LOCAL COPIES. No runtime
imports from V2/V3 are required.

BASIC SETTINGS  (edit policy/bangbang_coast_v4/settings.json)
-------------------------------------------------------------
  startup.mode = 'no_orient' | 'partial_turn' | 'full_turn'
               | 'rotate_and_thrust' | 'auto_compare'

  no_orient:
     NO initial reorientation. The body-frame direction used for thrust is
     set by startup.no_orient_body_selection:
       'target_los' = compute a body-frame oblique thrust request toward the
         nominal guidance aim (default; can be mostly +Z with small X/Y).
       'fixed' = exact startup.body_direction_B (e.g. [0,0,1]).
       'cone_search' = grid of body directions around the current nominal
         guidance direction, using startup.cone_offsets_deg. More CPU work.
     The attitude HOLD during the burn can still issue angular corrections;
     'no_orient' means no initial attitude alignment, NOT torque disabled.

  partial_turn:
     Start with body_direction_B; turn only startup.partial_fraction of the
     current-body-to-guidance direction angle (0 = none, 1 = full).
     Then thrust along the selected BODY axis.

  full_turn:
     Turn so startup.body_direction_B points in nominal guidance direction;
     then thrust. For legacy +X-like behavior set [1,0,0].

  rotate_and_thrust:
     Turn startup.combined_turn_fraction of the requested angle, AND
     request startup.turn_translation_fraction * burn.request_x_m_s2 along
     the chosen body axis during attitude alignment.
     Translation and angular requests share physical thrusters; combining
     them can reduce delivered authority and change fuel cost.

  auto_compare:
     Nominally compare no_orient, partial_turn, full_turn and
     rotate_and_thrust according to startup.auto_modes, with partial
     fraction candidates from startup.auto_partial_fractions.
     This is an approximate model ranking. The model has known errors.

  startup.body_direction_B: nonzero [x,y,z] BODY VECTOR, not necessarily
       a principal axis. The code normalizes the vector.
  startup.search_body_candidates: enable manual body_candidates_B sweep for
       full/partial/combined modes. Off by default to limit runtime cost.
  startup.required_prediction_feasible: true by default; on an infeasible
       nominal plan, stop BEFORE issuing the unverified main burn. For
       deliberate exploration a user may set false, but the resulting
       fallback duration has no capture guarantee.
  startup.candidate_burn_min_s/max_s/samples: coarse search bounds.
  startup.capture_radius_m: nominal 42m constraint; official scoring uses
       50m. Model errors can exceed this margin.
  startup.fallback_burn_s: planned duration only if model search is
       infeasible AND required_prediction_feasible=false.
  startup.online_burn: generic-body burn/coast duration reoptimization;
       uses latest observed public state. Independent of +X assumptions.

  launch.enabled: false by default. Enabling sends the configured boost
       BEFORE target observation and BEFORE V4 strategy selection; this
       is no longer the original V2/V3 initial boost by default.

  burn.joint_optimizer.enabled: MUST remain false. The inherited V3 joint
       optimizer assumes +X plus Y/Z and is NOT suitable for arbitrary
       body axes. V4 uses its own generic-axis online burn predictor.
       V4 does not yet implement proactive joint optimal lateral pulses.

  coast.mode: MUST remain 'ballistic'; the older corrected-coast mode
       assumes a +X burn frame and is rejected by V4.

  attitude / avoidance / coast: retain V3-local copies. The radar dodge
       is projected perpendicular to the SELECTED BODY thrust axis.
       Radar avoids during alignment and powered flight. Coast radar is
       MONITOR-ONLY, not active avoidance. Recovery may use the V3-derived
       +X fallback if no economical body pulse achieves interception.

  objective: no terminal translational velocity penalty. The capture point
       is free inside the sphere. Predicted first sphere entry is scored.

MODE PRESETS
------------
Complete configs are provided in presets/settings_<mode>.json.
Copy one to settings.json to change modes, inspect settings, then reselect
V4 using the runner so source/config hashes match the selected run.
The default 'no_orient' preset uses target_los selection, not pure +Z.

POWERSHELL (from STU4-Solution; simulator CLOSED)
-------------------------------------------------
  & $py -m unittest discover -s policy/bangbang_coast_v4/tests -v
  if ($LASTEXITCODE -ne 0) { throw 'V4 tests failed' }
  & $py .\run_simulation.py select --config 'policy/bangbang_coast_v4/run_plan.json'
  if ($LASTEXITCODE -ne 0) { throw 'V4 select failed' }
  & $py .\run_simulation.py check

Launch the official unchanged simulator manually using your existing
approved spectator workflow. Do not run automation or modify sim files.

Return to the verified V2 U2:
  & $py .\run_simulation.py select --config 'policy/bangbang_coast_v2/run_plan.json'

For a new V4 mode after closing the simulator:
  Copy-Item .\policy\bangbang_coast_v4\presets\settings_partial_turn.json `
    .\policy\bangbang_coast_v4\settings.json -Force
  & $py .\run_simulation.py select --config 'policy/bangbang_coast_v4/run_plan.json'
  & $py .\run_simulation.py check

RECORDING AND CAUSALITY
-----------------------
Run files are saved under policy/bangbang_coast_v4/run_logs. Events include
  v4_strategy_selected, v4_alignment_skipped, v4_alignment_started,
  v4_alignment_completed, v4_post_turn_refit, v4_burn_started,
  v4_burn_forecast, v4_burn_ended, coast_intercept_forecast,
  coast_direct_correction_selected, v4_recovery_plus_x_fallback.
The runner records public commands and states independently.

LIMITATIONS
-----------
- No exact attitude/translational joint switching-time optimum is solved
  online. Alignment uses V2-style closed-loop bang/off/bang switching;
  startup turn duration is approximate in preflight optimization.
- No guaranteed collision avoidance. Radar has noisy, delayed relative
  measurements without full debris velocity or persistent object IDs.
- During coast radar is passive; new collisions can occur. This is a known
  reliability problem and not solved by changing startup direction.
- Model mismatch (especially attitude during long coast) can make
  predicted 42m capture miss the official 50m zone.
- Generic-body online burn optimization assumes frozen measured attitude
  and nominal mean allocator. Actual actuator and rotation are coupled.
- Mode names are actual distinct executable strategies, not just labels;
  each logs its selected body vector and requested commands.
- Native performance unknown. Preserve the 47.7-point verified V2 policy.
