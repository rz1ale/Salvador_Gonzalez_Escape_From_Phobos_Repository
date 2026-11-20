"""Thin adapter for the existing run(context) interface; public I/O only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .radar import RadarTracker, load_settings, VERSION


class RadarStream:
    """One instance per native attempt. poll() returns the common in-memory frame.

    SDK calls stay on the caller/controller thread. No commands, simulation
    lifecycle actions, camera access, background SDK connections or private reads.
    """
    def __init__(self, context, settings=None, *, log_full_frames=False):
        self.context = context
        if settings is not None and not isinstance(settings, dict):
            root = Path(context.root).resolve()
            source = Path(settings)
            source = source if source.is_absolute() else root/source
            if not source.resolve().is_relative_to(root):
                raise ValueError('Radar settings must be inside this solution clone')
            settings = source
        self.settings = load_settings(settings)
        self.tracker = RadarTracker(self.settings)
        self.log_full_frames = bool(log_full_frames)
        self.latest = None
        self._last_poll = self._last_log = -float('inf')
        self.output = Path(context.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        sources = ('__init__.py','radar.py','collision.py','runtime.py')
        self._append('obstacle_diagnostics.txt', dict(event='initialized', producer_version=VERSION,
            settings=self.settings,
            settings_sha256=hashlib.sha256(json.dumps(self.settings, sort_keys=True).encode()).hexdigest(),
            source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
            SDK_calls='context.state/radar/clock only', full_frame_logging=self.log_full_frames))

    def _append(self, filename, value):
        with (self.output/filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False)+'\n')

    def poll(self, own_state=None, *, force=False):
        """Pass an already read context.state() packet to avoid a duplicate RPC."""
        packet = self.context.state() if own_state is None else own_state
        self.tracker.add_own_state(packet)
        now = self.context.clock()
        if self.latest is not None and not force and now-self._last_poll < self.settings['poll_period_s']:
            return self.latest
        rows = self.context.radar(500)
        received = self.context.clock()
        frame = self.tracker.update(rows, received)
        self.latest = frame
        self._last_poll = received
        if received-self._last_log >= self.settings['diagnostic_log_period_s']:
            self._append('obstacle_diagnostics.txt', dict(time_sim_s=received, sequence=frame['sequence'],
                clock=frame['clock'], coverage=frame['coverage'], diagnostics=frame['diagnostics']))
            self._last_log = received
        if self.log_full_frames:
            self._append('obstacle_stream.txt', frame)
        return frame
