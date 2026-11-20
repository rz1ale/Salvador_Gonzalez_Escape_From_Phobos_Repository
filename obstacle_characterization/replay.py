"""Offline replay of existing public states.txt/radar.txt; never starts a sim.

Run from solution root: python -m obstacle_characterization.replay --session PATH --show
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .radar import RadarTracker, load_settings, VERSION

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_attempt(attempt, settings=None, frame_path=None):
    attempt = Path(attempt)
    own_file, radar_file = attempt/'states.txt', attempt/'radar.txt'
    tracker = RadarTracker(settings)
    own = iter(records(own_file))
    next_own = next(own, None)
    elapsed, objects, confirmed = [], [], []
    totals = {}
    first = last = None
    last_radar = -float('inf')
    frame = None
    output = Path(frame_path).open('w', encoding='utf-8') if frame_path else None
    try:
        for packet in records(radar_file):
            clock = float(packet['read_sim_s'])
            if clock < last_radar-1e-8:
                raise ValueError('Radar receive clock moved backwards')
            last_radar = clock
            while next_own is not None and .5*(next_own['read_begin_sim_s']+next_own['read_end_sim_s']) <= clock:
                tracker.add_own_state(next_own)
                next_own = next(own, None)
            # No future own state or future radar packet is exposed to tracker.
            frame = tracker.update(packet['observations'], clock)
            if output:
                output.write(json.dumps(frame, allow_nan=False)+'\n')
            elapsed.append(frame['diagnostics']['processing_wall_ms'])
            objects.append(len(frame['objects']))
            confirmed.append(sum(o['status'] in ('confirmed','coasting') for o in frame['objects']))
            for key, value in frame['diagnostics'].items():
                if type(value) is int:
                    totals[key] = totals.get(key, 0)+value
            first = clock if first is None else first
            last = clock
    finally:
        if output:
            output.close()
    if frame is None:
        raise ValueError('No recorded radar packets')
    return dict(attempt=str(attempt), radar_frames=len(elapsed), first_sim_s=first, last_sim_s=last,
        counters=totals, maximum_output_objects=max(objects), maximum_confirmed_tracks=max(confirmed),
        final_clock=frame['clock'], final_coverage=frame['coverage'],
        processing_wall_ms=dict(p50=float(np.percentile(elapsed, 50)), p95=float(np.percentile(elapsed, 95)),
            maximum=float(max(elapsed)), total=float(sum(elapsed))),
        input_sha256=dict(states=digest(own_file), radar=digest(radar_file)),
        note='Public-record replay. Timing excludes native SDK I/O and optional JSON frame serialization. No debris-truth accuracy/coverage claim.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--attempt')
    select.add_argument('--session')
    parser.add_argument('--settings')
    parser.add_argument('--out')
    parser.add_argument('--frames', action='store_true', help='Write full standardized frames as .txt JSON Lines; can be large')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    root = ROOT.resolve()
    def local(value):
        p = Path(value)
        p = (p if p.is_absolute() else root/p).resolve()
        if not p.is_relative_to(root):
            raise ValueError('CLI paths must stay in this solution clone')
        return p
    source = local(args.attempt or args.session)
    settings = load_settings(local(args.settings) if args.settings else None)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = local(args.out) if args.out else root/'obstacle_characterization/analysis_results'/source.name/('replay_'+stamp)
    output.mkdir(parents=True, exist_ok=False)
    if args.session:
        ledger = json.loads((source/'ledger.json').read_text(encoding='utf-8-sig'))
        attempts = []
        for row in ledger['attempts']:
            path = (source/row['directory']).resolve()
            if not path.is_relative_to(source):
                raise ValueError('Attempt directory escapes session')
            attempts.append((row['trial_id']+'__'+row['attempt_id'], path))
    else:
        attempts = [(source.name, source)]
    results = []
    for name, path in attempts:
        if not (path/'states.txt').is_file() or not (path/'radar.txt').is_file():
            results.append(dict(attempt=str(path), skipped='states.txt or radar.txt missing'))
            continue
        print('Public radar replay:', name, flush=True)
        safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
        result = replay_attempt(path, settings, output/(safe_name+'_stream.txt') if args.frames else None)
        results.append(result)
    metrics = dict(schema_version=1, producer_version=VERSION, settings=settings,
        source=str(source), environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        results=results, independent_obstacle_truth_available=False, native_SDK_latency_measured=False)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    lines = ['RADAR CHARACTERIZATION / PUBLIC-RECORD REPLAY', 'Source: '+str(source),
        'attempt\tframes\tmax_objects\tmax_confirmed\tprocessing_p50_ms\tprocessing_p95_ms']
    for result in results:
        if 'skipped' in result:
            lines.append(result['attempt']+'\tSKIPPED: '+result['skipped'])
        else:
            lines.append('\t'.join(map(str, (result['attempt'], result['radar_frames'],
                result['maximum_output_objects'], result['maximum_confirmed_tracks'],
                result['processing_wall_ms']['p50'], result['processing_wall_ms']['p95']))))
    lines += ['', 'No simulation launched or changed. No private obstacle data used.',
        'Track identities/motion are estimates. This report is not ground-truth detection accuracy.',
        'Offline processing timings exclude native I/O; measure actual controller deadlines separately.',
        'Full frame logs are optional (--frames); existing raw recordings remain unchanged.']
    text = '\n'.join(lines)+'\n'
    (output/'summary.txt').write_text(text, encoding='utf-8')
    if args.show:
        print(text)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
