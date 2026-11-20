"""NumPy-only runtime reference + declared augmented TVLQR error feedback."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .math_core import (sha256,signature,settings,matrix,error,interpolate,clip,direction_rotation)


def packet_row(packet):
    q=np.asarray(packet["q_PB_xyzw_reported"],float)
    return np.r_[packet["position_P_m"],packet["velocity_P_m_s"],q,
        matrix(q)@np.asarray(packet["omega_relP_P_rad_s"],float)]


def causal_state(times,rows,p):
    t=np.asarray(times);y=np.asarray(rows);x=y[-1].copy();use=t>=t[-1]-.3
    for sl,bias,delay in ((slice(3,6),p["velocity_report_bias_P_m_s"],p["velocity_report_delay_s"]),
            (slice(10,13),p["rate_report_bias_B_rad_s"],p["rate_report_delay_s"])):
        slope=np.linalg.lstsq(np.c_[t[use]-t[-1],np.ones(use.sum())],y[use,sl],rcond=None)[0][0] if use.sum()>=3 else np.zeros(3)
        x[sl]+=float(delay)*slope-np.asarray(bias)
    x[:3]-=matrix(x[6:10]).T@np.asarray(p["reporting_offset_B_m"])
    return x


class Policy:
    def __init__(self,folder,model_path):
        self.folder=Path(folder);self.s=settings(self.folder/"settings.json")
        path=self.folder/"assets"/"manifest.json"
        if not path.is_file():raise ValueError("No prepared trajectory. Run precompute.py --select before opening the simulator.")
        self.manifest=json.loads(path.read_text(encoding="utf-8-sig"))
        if self.manifest.get("schema_version")!=1 or not self.manifest.get("export_verified"):
            raise ValueError("Unverified trajectory; run precompute.py --select")
        if (signature(self.s)!=self.manifest["design_signature"] or sha256(model_path)!=self.manifest["model_sha256"] or
            sha256(self.folder/"mission.json")!=self.manifest["mission_sha256"]):
            raise ValueError("Settings/model changed. Close SpaceCRAFT, then run: & $py .\\policy\\nonlinear_handoff_staging\\precompute.py --select")
        for name,digest in self.manifest["runtime_sources_sha256"].items():
            if sha256(self.folder/name)!=digest:raise ValueError("Policy source changed; regenerate offline: "+name)
        asset=self.folder/"assets"/"reference.npz"
        if sha256(asset)!=self.manifest["reference_sha256"]:raise ValueError("Corrupt trajectory cache")
        with np.load(asset,allow_pickle=False) as d:
            for key in ("times","states","command_times","commands","gain_times","gains","input_scales","continuation_gain"):
                setattr(self,key,np.asarray(d[key],float))
        if (self.states.shape!=(len(self.times),13) or self.commands.shape!=(len(self.command_times),6) or
            self.gains.shape!=(len(self.gain_times),6,18) or self.input_scales.shape!=(6,) or
            not all(np.isfinite(a).all() for a in (self.states,self.commands,self.gains,self.input_scales)) or
            np.any(np.diff(self.times)<=0) or len(self.times)<2):raise ValueError("Invalid reference shapes/time grid")
        self.definition=json.loads(Path(model_path).read_text(encoding="utf-8-sig"))
        self.parameters=self.definition["parameters"];self.caps=self.definition["actuator"]["request_norm_caps"]
        self.duration=float(self.s["design"]["duration_s"]);self.xf=np.asarray(self.manifest["reference"]["xf"])
        self.pd=np.asarray(self.manifest["reference"]["pd"]);self.n=np.asarray(self.manifest["reference"]["n"])
        self.terminal_input=np.asarray(self.manifest["terminal_public_request"])
        self.angular_input_map=float(self.parameters["mean_native_period_s"])*np.linalg.inv(self.parameters["angular_gain"])

    def nominal_command(self,t):
        i=min(len(self.commands)-1,max(0,int(np.searchsorted(self.command_times,t+1e-9,side="right")-1)))
        return self.commands[i].copy()

    def control(self,t,x,previous):
        if t<self.duration-1e-9:
            ref=interpolate(self.times,self.states,t);ff=self.nominal_command(t)
            j=min(len(self.gains)-1,max(0,int(np.searchsorted(self.gain_times,t+1e-9,side="right")-1)))
            old_ref=self.nominal_command(t-self.s["feedback"]["period_s"]) if t>=self.s["feedback"]["period_s"] else np.zeros(6)
            augmented=np.r_[error(x,ref),(np.asarray(previous)-old_ref)/self.input_scales]
            correction=-self.gains[j]@augmented
            largest=float(np.max(abs(correction)));limit=self.s["feedback"]["maximum_normalized_correction"]
            if largest>limit:correction*=limit/largest
            u=clip(ff+self.input_scales*correction,self.caps)
            kind="staging_trajectory"
        else:
            # Declared continuation: stop tracking staging POSITION at the endpoint.
            # Maintain full forward request + pointing/rate feedback; no NLP online.
            p=x[:3]+matrix(x[6:10]).T@np.asarray(self.parameters["reporting_offset_B_m"])
            axis,angle=direction_rotation(matrix(x[6:10])@(self.xf-p))
            angle_error=axis*angle
            if angle<=np.radians(self.s["feedback"]["continuation_pointing_deadband_deg"]):angle_error[:]=0
            omega=x[10:13].copy()
            if np.linalg.norm(omega)<=self.s["feedback"]["continuation_rate_deadband_rad_s"]:omega[:]=0
            kp,kd=self.continuation_gain;alpha=kp*angle_error-kd*omega
            length=np.linalg.norm(alpha);limit=self.s["feedback"]["continuation_alpha_limit_rad_s2"]
            if length>limit:alpha*=limit/length
            u=np.r_[self.terminal_input[:3],self.angular_input_map@alpha]
            factors=[cap/np.linalg.norm(u[sl]) for sl,cap in ((slice(0,3),self.caps[0]),(slice(3,6),self.caps[1])) if np.linalg.norm(u[sl])>1e-12]
            u=clip(u*max(1.,min(factors)),self.caps)
            kind="forward_continuation";ref=self.states[-1]
        pr=x[:3]+matrix(x[6:10]).T@np.asarray(self.parameters["reporting_offset_B_m"])
        _,angle=direction_rotation(matrix(x[6:10])@(self.xf-pr))
        return u,dict(stage=kind,pointing_error_deg=float(np.degrees(angle)),
            rate_norm_rad_s=float(np.linalg.norm(x[10:13])),reference_error=error(x,ref).tolist(),
            xd_distance_m=float(np.linalg.norm(pr-self.pd)),forward_velocity_m_s=float(self.n@x[3:6]))
