"""Flight math and precomputed controller: NumPy only, no SDK/LP/solver imports."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_settings(path):
    s = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    expected = {"schema_version", "label", "boost_duration_s", "boost_thrust_fraction",
        "slew_duration_s", "slew_steps", "slew_weights", "tracking_weights", "forward_thrust_fraction",
        "hold_weights", "feedback_acceleration_limit_rad_s2", "hold_pointing_deadband_deg",
        "hold_rate_deadband_rad_s", "control_period_s", "test_duration_s", "prediction_duration_s",
        "status_period_s"}
    if set(s) != expected or s["schema_version"] != 1:
        raise ValueError("Unsupported/unknown policy settings")
    for k in ("boost_duration_s", "slew_duration_s", "control_period_s", "prediction_duration_s", "status_period_s"):
        if not isinstance(s[k], (int, float)) or not np.isfinite(s[k]) or s[k] <= 0:
            raise ValueError("Expected positive " + k)
    if type(s["slew_steps"]) is not int or not 2 <= s["slew_steps"] <= 1000:
        raise ValueError("slew_steps must be an integer in 2..1000")
    for k in ("boost_thrust_fraction", "forward_thrust_fraction"):
        if not 0 <= s[k] <= 1:
            raise ValueError(k + " must be in [0,1]")
    if s["test_duration_s"] is not None and (not np.isfinite(s["test_duration_s"]) or s["test_duration_s"] <= 0):
        raise ValueError("test_duration_s must be null or positive")
    for group, keys in (("slew_weights", {"angle", "rate", "effort", "fuel", "terminal_angle", "terminal_rate"}),
                        ("tracking_weights", {"angle", "rate", "effort", "terminal_angle", "terminal_rate"}),
                        ("hold_weights", {"angle", "rate", "effort"})):
        if set(s[group]) != keys or any(not np.isfinite(v) or v < 0 for v in s[group].values()) or s[group]["effort"] <= 0:
            raise ValueError("Invalid " + group)
    for k in ("feedback_acceleration_limit_rad_s2", "hold_pointing_deadband_deg", "hold_rate_deadband_rad_s"):
        if not np.isfinite(s[k]) or s[k] < 0:
            raise ValueError("Invalid " + k)
    return s


def matrix(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def multiply(a, b):
    return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3], b[:3]), a[3]*b[3]-a[:3]@b[:3]]


def increment(q, physical_rotation_B):
    v = -np.asarray(physical_rotation_B, float)
    angle = np.linalg.norm(v)
    dq = np.r_[v * (np.sin(angle/2)/angle if angle > 1e-10 else 0.5), np.cos(angle/2)]
    out = multiply(dq, q)
    return out / np.linalg.norm(out)


def rotation_needed(current, desired):
    """Physical shortest rotation from current to desired, in CURRENT body axes."""
    dq = multiply(desired, np.r_[-current[:3], current[3]])
    if dq[3] < 0:
        dq = -dq
    n = np.linalg.norm(dq[:3])
    return -dq[:3] * (2*np.arctan2(n, max(dq[3], 0.0))/n if n > 1e-10 else 2.0)


def direction_rotation(direction_B):
    d = np.asarray(direction_B, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    cross = np.cross([1.0, 0.0, 0.0], d)
    n = np.linalg.norm(cross)
    angle = float(np.arctan2(n, np.clip(d[0], -1, 1)))
    axis = cross / n if n > 1e-10 else np.array([0.0, 1.0, 0.0])
    return axis, angle


def closest_point(position, target, radius):
    delta = np.asarray(position) - np.asarray(target)
    n = np.linalg.norm(delta)
    if n <= radius:
        return np.asarray(position, float).copy()
    return np.asarray(target) + radius * delta / n


def pointing_error(q, position, goal):
    d = np.asarray(goal) - np.asarray(position)
    if np.linalg.norm(d) < 1e-9:
        return 0.0
    return direction_rotation(matrix(q) @ d)[1]


class Policy:
    def __init__(self, folder, model_path, *, verify=True):
        self.folder = Path(folder)
        self.settings = load_settings(self.folder / "settings.json")
        self.manifest = json.loads((self.folder / "assets" / "manifest.json").read_text(encoding="utf-8-sig"))
        asset = self.folder / "assets" / "reference.npz"
        if verify:
            for path, key in ((model_path, "model_sha256"), (self.folder / "settings.json", "settings_sha256"),
                              (self.folder / "mission.json", "mission_sha256"), (asset, "reference_sha256")):
                if sha256(path) != self.manifest[key]:
                    raise ValueError("Offline reference differs from " + str(path) + "; run precompute.py again")
            for name,digest in self.manifest.get("generator_sources_sha256",{}).items():
                if sha256(self.folder/name)!=digest:
                    raise ValueError("Reference generator changed: "+name+"; run precompute.py again")
        with np.load(asset, allow_pickle=False) as d:
            self.times = d["slew_times"].copy()
            self.rotation = d["slew_states"].copy()
            self.alpha = d["slew_alpha"].copy()
            self.gains = d["tracking_gains"].copy()
            self.hold_gain = d["hold_gain"].copy()
        n = self.settings["slew_steps"]
        arrays = (self.times,self.rotation,self.alpha,self.gains,self.hold_gain)
        if (self.times.shape != (n+1,) or self.rotation.shape != (n+1,2) or
            self.alpha.shape != (n,) or self.gains.shape != (n,2) or self.hold_gain.shape != (2,) or
            not all(np.isfinite(a).all() for a in arrays) or np.any(np.diff(self.times)<=0)):
            raise ValueError("Invalid offline reference arrays; run precompute.py again")
        definition = json.loads(Path(model_path).read_text(encoding="utf-8-sig"))
        self.input_map = float(definition["parameters"]["mean_native_period_s"]) * np.linalg.inv(definition["parameters"]["angular_gain"])
        self.rate_bias = np.asarray(definition["parameters"]["rate_report_bias_B_rad_s"])
        self.caps = np.asarray(definition["actuator"]["request_norm_caps"])
        self.x_knee = self.manifest["forward_request_at_saturation_knee"]
        self.z_knee = self.manifest["boost_request_at_saturation_knee"]
        self.started = False
        self.slew_started = False

    def start(self, packet):
        self.xf = closest_point(packet["position_P_m"], packet["target_P_m"], self.manifest["arrival_radius_m"])
        self.started = True

    def command(self, elapsed_s, packet):
        if not self.started:
            self.start(packet)
        s = self.settings
        q = np.asarray(packet["q_PB_xyzw_reported"], float)
        R = matrix(q)
        w = R @ np.asarray(packet["omega_relP_P_rad_s"], float) - self.rate_bias
        p = np.asarray(packet["position_P_m"], float)
        u = np.zeros(6)
        refq = q.copy()
        refw = np.zeros(3)
        angle_error = pointing_error(q, p, self.xf)
        feedback = np.zeros(3)
        if elapsed_s < s["boost_duration_s"]:
            stage = "boost"
            u[2] = self.z_knee * s["boost_thrust_fraction"]
        else:
            if not self.slew_started:
                self.anchor_q = q.copy()
                self.axis, self.angle = direction_rotation(R @ (self.xf-p))
                self.slew_started = True
            stage_t = elapsed_s - s["boost_duration_s"]
            if stage_t < s["slew_duration_s"]:
                stage = "slew"
                j = min(len(self.alpha)-1, max(0, int(np.searchsorted(self.times, stage_t, side="right")-1)))
                dt = stage_t - self.times[j]
                nominal = self.rotation[j]
                theta = nominal[0] + nominal[1]*dt + 0.5*self.alpha[j]*dt*dt
                rate = nominal[1] + self.alpha[j]*dt
                scale = self.angle / max(self.manifest["slew_angle_rad"], 1e-12)
                refq = increment(self.anchor_q, self.axis * theta * scale)
                transform = R @ matrix(refq).T
                refw = transform @ (self.axis * rate * scale)
                ff = transform @ (self.axis * self.alpha[j] * scale)
                error = rotation_needed(q, refq)
                kp, kd = self.gains[j]
                feedback = kp*error - kd*(w-refw)
                n = np.linalg.norm(feedback)
                limit = s["feedback_acceleration_limit_rad_s2"]
                if n > limit:
                    feedback *= limit/n
                u[3:] = self.input_map @ (ff + feedback)
                angle_error = float(np.linalg.norm(error))
            else:
                stage = "forward"
                axis, angle_error = direction_rotation(R @ (self.xf-p))
                kp, kd = self.hold_gain
                error = axis * angle_error
                if angle_error <= np.radians(s["hold_pointing_deadband_deg"]):
                    error[:] = 0.0
                wd = w.copy()
                if np.linalg.norm(wd) <= s["hold_rate_deadband_rad_s"]:
                    wd[:] = 0.0
                feedback = kp*error - kd*wd
                n = np.linalg.norm(feedback)
                limit = s["feedback_acceleration_limit_rad_s2"]
                if n > limit:
                    feedback *= limit/n
                u[0] = self.x_knee * s["forward_thrust_fraction"]
                u[3:] = self.input_map @ feedback
        # Full-thrust phases use a LARGE common scaling, preserving all six
        # request ratios. The native LP is positively homogeneous; once pulse
        # normalization saturates this leaves mean delivery unchanged, while
        # avoiding reliance on a fitted .021 s tick to guarantee saturation.
        # Scaling each request group independently would destroy the desired
        # translation/rotation ratio. Partial-thrust settings retain their
        # mean-model amplitude, because amplifying them would change delivery.
        full = ((stage == "boost" and s["boost_thrust_fraction"] == 1.0) or
                (stage == "forward" and s["forward_thrust_fraction"] == 1.0))
        if full:
            factors = [cap/np.linalg.norm(u[sl]) for sl,cap in
                ((slice(0,3),self.caps[0]),(slice(3,6),self.caps[1])) if np.linalg.norm(u[sl])>1e-12]
            u *= max(1.0,min(factors))
        for sl, cap in ((slice(0,3), self.caps[0]), (slice(3,6), self.caps[1])):
            n = np.linalg.norm(u[sl])
            if n > cap:
                u[sl] *= cap/n
        return u, dict(stage=stage, pointing_error_deg=float(np.degrees(pointing_error(q, p, self.xf))),
            tracking_error_deg=float(np.degrees(angle_error)), omega_B_rad_s=w.tolist(),
            reference_omega_B_rad_s=refw.tolist(), feedback_alpha_B_rad_s2=feedback.tolist(),
            closest_reference_P_m=self.xf.tolist())
