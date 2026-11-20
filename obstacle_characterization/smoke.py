"""Optional observation-only task for the existing runner. No vehicle commands."""
import json
from obstacle_characterization import RadarStream


def run(context):
    duration = float(context.parameters.get('duration_s', 4.))
    if not 1 <= duration <= 10:
        raise ValueError('Smoke duration must be 1..10 simulation seconds')
    context.request_stop_after_task()
    stream = RadarStream(context)
    start = context.clock()
    sequences = set()
    maximum = 0
    while context.clock()-start < duration:
        frame = stream.poll()
        json.dumps(frame, allow_nan=False)
        sequences.add(frame['sequence'])
        maximum = max(maximum, len(frame['objects']))
        context.sleep(.05)
    passed = len(sequences)>=2 and maximum>0
    context.finish(outcome='success' if passed else 'failure',
        reason='obstacle_stream_smoke_complete' if passed else 'obstacle_stream_not_assessed',
        frames=len(sequences), maximum_output_objects=maximum,
        official_score_captured=False, collision_safety_assessed=False)
