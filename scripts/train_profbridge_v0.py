from __future__ import annotations
import argparse, json, math, random, sys, time
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Iterable
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from profbridge.data.jsonl import read_jsonl, write_jsonl
from profbridge.utils.env import ensure_dir, get_git_commit, utc_timestamp

SUCCESS = {"success", "partial_success"}
KEY_FIELDS = ["task_id", "candidate_source_type", "generation_method", "source_hash", "candidate_config_hash"]
SANITY_PREFIXES = ("gpu__time_duration", "gpu__time_active")
MAIN_PREFIXES = ("dram__bytes", "dram__bytes_read", "dram__bytes_write", "sm__warps_active", "smsp__cycles_active", "smsp__cycles_elapsed", "smsp__inst_executed")
V2_HINTS = ("smsp__warp_issue_stalled", "smsp__warps_eligible", "smsp__average_warps", "l1tex__", "local")
MAIN_PREFIXES = ("dram__bytes", "dram__bytes_read", "dram__bytes_write", "sm__warps_active", "smsp__cycles_active", "smsp__cycles_elapsed", "smsp__inst_executed")
V2_HINTS = ("smsp__warp_issue_stalled", "smsp__warps_eligible", "smsp__average_warps", "l1tex__", "local")
MAIN_PREFIXES = ("dram__bytes", "dram__bytes_read", "dram__bytes_write", "sm__warps_active", "smsp__cycles_active", "smsp__cycles_elapsed", "smsp__inst_executed")
V2_HINTS = ("smsp__warp_issue_stalled", "smsp__warps_eligible", "smsp__average_warps", "l1tex__", "local")
IDENTITY_KEYS = {"source_hash", "candidate_config_hash", "candidate_key", "candidate_id", "candidate_source_path", "candidate_path", "raw_latency_file", "unique_key"}
FEATURE_SETS = ["F0", "F1", "F2", "F3", "F4", "F5"]


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    out: dict[str, Path] = {}
    for pat in patterns:
        matches = [Path(p) for p in glob(pat)]
        p = Path(pat)
        if not matches and p.exists():
            matches = [p]
        for m in matches:
            if m.is_file():
                out[str(m)] = m
    return [out[k] for k in sorted(out)]


def metrics_of(r: dict[str, Any]) -> dict[str, float]:
    prof = r.get("high_fidelity_profile") if isinstance(r.get("high_fidelity_profile"), dict) else {}
    raw = prof.get("selected_ncu_metrics") or prof.get("parsed_metrics") or {}
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[str(k)] = float(v)
        elif isinstance(v, dict) and isinstance(v.get("mean"), (int, float)) and math.isfinite(float(v["mean"])):
            out[str(k)] = float(v["mean"])
    return out



def metric_profile_version(r: dict[str, Any]) -> str:
    prof = r.get("high_fidelity_profile") if isinstance(r.get("high_fidelity_profile"), dict) else {}
    version = r.get("ncu_metric_profile_version") or prof.get("ncu_metric_profile_version")
    if version:
        return str(version)
    metrics = metrics_of(r)
    if any(k.startswith(("smsp__warp_issue_stalled", "l1tex__")) or "local" in k for k in metrics):
        return "phase05b_v2_inferred"
    return "legacy_or_default_v1"

def missing_keys(r: dict[str, Any]) -> list[str]:
    return [k for k in KEY_FIELDS if r.get(k) in {None, ""}]


def usable_reason(r: dict[str, Any]) -> tuple[bool, str]:
    if not r.get("correctness_pass"):
        return False, "correctness_failed"
    prof = r.get("high_fidelity_profile") if isinstance(r.get("high_fidelity_profile"), dict) else {}
    if prof.get("profiling_status") not in SUCCESS:
        return False, f"profile_status_{prof.get('profiling_status')}"
    if not metrics_of(r):
        return False, "no_numeric_metrics"
    lat = r.get("latency_stats") if isinstance(r.get("latency_stats"), dict) else {}
    if not isinstance(lat.get("mean_ms"), (int, float)):
        return False, "missing_latency_mean_ms"
    miss = missing_keys(r)
    if miss:
        return False, "missing_key_fields:" + ",".join(miss)
    return True, "usable"


def rec_ts(r: dict[str, Any]) -> str:
    env = r.get("environment") if isinstance(r.get("environment"), dict) else {}
    return str(env.get("timestamp") or r.get("timestamp") or r.get("experiment_id") or "")


def load_dataset(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records, errors = [], []
    for p in paths:
        try:
            for i, r in enumerate(read_jsonl(p)):
                r = dict(r); r["_manifest_file"] = str(p); r["_manifest_index"] = i; records.append(r)
        except Exception as e:
            errors.append({"path": str(p), "error": f"{type(e).__name__}: {e}"})
    reasons, usable = Counter(), []
    for r in records:
        ok, why = usable_reason(r)
        if ok: usable.append(r)
        else: reasons[why] += 1
    audit = {
        "input_files": [str(p) for p in paths],
        "input_raw_record_count": len(records),
        "predictor_usable_record_count": len(usable),
        "excluded_record_count": len(records)-len(usable),
        "excluded_reasons": dict(sorted(reasons.items())),
        "read_errors": errors,
        "task_id_distribution": dict(sorted(Counter(str(r.get("task_id")) for r in usable).items(), key=lambda x: x[0])),
        "candidate_source_type_distribution": dict(sorted(Counter(str(r.get("candidate_source_type")) for r in usable).items())),
        "generation_method_distribution": dict(sorted(Counter(str(r.get("generation_method")) for r in usable).items())),
        "manual_generated_count": sum(1 for r in usable if r.get("candidate_source_type") in {"manual_seed_candidate", "generated_candidate"}),
        "real_llm_generated_count": sum(1 for r in usable if r.get("candidate_source_type") == "future_llm_generated_candidate"),
        "metric_version_distribution": dict(sorted(Counter(metric_profile_version(r) for r in usable).items())),
        "missing_key_field_summary_all_records": dict(sorted(Counter(k for r in records for k in missing_keys(r)).items())),
        "complete_key_record_count": sum(1 for r in records if not missing_keys(r)),
        "complete_key_record_count_usable": sum(1 for r in usable if not missing_keys(r)),
        "chronological_timestamp_count": sum(1 for r in usable if rec_ts(r)),
        "task_count": len({r.get("task_id") for r in usable}),
        "candidate_source_type_count": len({r.get("candidate_source_type") for r in usable}),
    }
    audit["leave_task_out_meaningful"] = audit["task_count"] >= 5 and len(usable) >= 30
    audit["chronological_split_meaningful"] = audit["chronological_timestamp_count"] >= max(10, int(0.8*len(usable)))
    audit["candidate_source_holdout_meaningful"] = audit["candidate_source_type_count"] >= 3 and len(usable) >= 60
    return usable, audit


def add_cat(f: dict[str, float], k: str, v: Any) -> None:
    f[f"{k}={v if v not in {None, ''} else '<missing>'}"] = 1.0


def flat(v: Any, pref: str, f: dict[str, float]) -> None:
    if v is None:
        f[pref + ".missing"] = 1.0
    elif isinstance(v, bool):
        f[pref] = 1.0 if v else 0.0
    elif isinstance(v, (int, float)) and math.isfinite(float(v)):
        x = float(v); f[pref] = x
        if x >= 0: f[pref + ".log1p"] = math.log1p(x)
    elif isinstance(v, str):
        if v and len(v) <= 96: add_cat(f, pref, v)
    elif isinstance(v, dict):
        for k, vv in v.items():
            if str(k) not in IDENTITY_KEYS: flat(vv, f"{pref}.{k}" if pref else str(k), f)
    elif isinstance(v, (list, tuple)):
        for i, vv in enumerate(v[:16]): flat(vv, f"{pref}.{i}" if pref else str(i), f)


def features(r: dict[str, Any], fs: str, include_task_id: bool = True) -> dict[str, float]:
    f: dict[str, float] = {}
    if include_task_id:
        add_cat(f, "task_id", r.get("task_id"))
    add_cat(f, "candidate_source_type", r.get("candidate_source_type")); add_cat(f, "generation_method", r.get("generation_method"))
    if fs in {"F1","F2","F3","F4","F5"}:
        lat = r.get("latency_stats") if isinstance(r.get("latency_stats"), dict) else {}
        for k in ["mean_ms","median_ms","std_ms","p05_ms","p95_ms","num_warmup","num_repeats"]: flat(lat.get(k), f"latency.{k}", f)
        if isinstance(lat.get("mean_ms"), (int,float)) and isinstance(lat.get("std_ms"), (int,float)) and abs(float(lat["mean_ms"])) > 1e-12:
            f["latency.cv"] = float(lat["std_ms"])/abs(float(lat["mean_ms"]))
    if fs in {"F2","F3","F4","F5"}:
        cheap = r.get("cheap_features") if isinstance(r.get("cheap_features"), dict) else {}
        for k,v in cheap.items():
            if k not in IDENTITY_KEYS and k != "static_feature_time_sec": flat(v, f"cheap.{k}", f)
    if fs in {"F3","F4","F5"}:
        cheap = r.get("cheap_features") if isinstance(r.get("cheap_features"), dict) else {}
        for k,v in cheap.items():
            if any(t in str(k).lower() for t in ["ptxas","compile","register","shared","spill","local_memory","binary","object"]): flat(v, f"compiler.{k}", f)
        cost = r.get("cost_accounting") if isinstance(r.get("cost_accounting"), dict) else {}
        for k in ["compile_time_sec","static_feature_time_sec"]: flat(cost.get(k), f"cost.{k}", f)
    if fs in {"F4","F5"}:
        for k in ["backend","is_baseline_candidate","is_generated_candidate","is_manual_seed","oom_timeout_risk"]:
            v = r.get(k); add_cat(f, f"meta.{k}", v)
        f["meta.has_backend_config_hash"] = 1.0 if r.get("backend_config_hash") else 0.0
        f["meta.has_source_hash"] = 1.0 if r.get("source_hash") else 0.0
        f["meta.has_candidate_config_hash"] = 1.0 if r.get("candidate_config_hash") else 0.0
        hist = r.get("transformation_history") if isinstance(r.get("transformation_history"), list) else []
        for h in hist[:16]: add_cat(f, "history", h)
        notes = str(r.get("notes") or "").lower()
        for tok in ["pytorch","primitive","inline cuda","reference","fallback","manual","generated"]:
            if tok in notes: f["notes.has_"+tok.replace(" ","_")] = 1.0
    if fs == "F5":
        add_cat(f, "metric_profile_version", metric_profile_version(r))
    return f

class Vec:
    def __init__(self): self.names=[]; self.idx={}; self.mean=None; self.scale=None
    def fit(self, rows):
        self.names = sorted({k for r in rows for k in r}); self.idx={k:i for i,k in enumerate(self.names)}
        x = self.raw(rows); self.mean = x.mean(0) if x.size else np.zeros(len(self.names)); self.scale = x.std(0) if x.size else np.ones(len(self.names)); self.scale[self.scale < 1e-12] = 1.0; return self
    def raw(self, rows):
        x = np.zeros((len(rows), len(self.names)))
        for i,r in enumerate(rows):
            for k,v in r.items():
                j=self.idx.get(k)
                if j is not None and math.isfinite(float(v)): x[i,j]=float(v)
        return x
    def trans(self, rows):
        x=self.raw(rows); return (x-self.mean)/self.scale if self.mean is not None else x


def target_summary(records):
    vals=defaultdict(list); versions=defaultdict(Counter)
    for r in records:
        ver=metric_profile_version(r)
        for k,v in metrics_of(r).items():
            vals[k].append(v); versions[k][ver]+=1
    out={}; n=len(records)
    for k,vs in sorted(vals.items()):
        a=np.asarray(vs,float); mean=float(a.mean()); std=float(a.std()); cov=len(vs)/n if n else 0.0; unique=len(set(float(v) for v in vs)); cv=std/abs(mean) if abs(mean)>1e-12 else None
        group="excluded"; selected=False; reason=None
        if unique <= 1 or std <= 1e-12:
            reason="near_zero_variance"
        elif k.startswith(SANITY_PREFIXES):
            group="sanity"; selected=cov>=0.70; reason=None if selected else "sanity_coverage_below_70pct"
        elif any(k.startswith(p) for p in MAIN_PREFIXES):
            group="main"; selected=cov>=0.70; reason=None if selected else "main_coverage_below_70pct"
        elif any(h in k for h in V2_HINTS) or any("phase05b_v2" in v for v in versions[k]):
            group="v2_supplementary"; selected=False; reason="supplementary_low_coverage_or_version_specific"
        elif cov >= 0.70:
            group="main"; selected=True
        else:
            reason="coverage_below_70pct"
        out[k]={"coverage":cov,"count":len(vs),"mean":mean,"std":std,"cv":cv,"min":float(a.min()),"max":float(a.max()),"unique_values":unique,"group":group,"metric_group":group,"selected":selected,"excluded_reason":reason,"metric_profile_version_counts":dict(sorted(versions[k].items()))}
    return out

def split_records(records, name):
    if name == "random":
        rows=list(records); random.Random(0).shuffle(rows); cut=max(1,min(len(rows)-1,int(.7*len(rows)))); return [(rows[:cut], rows[cut:], "random")]
    if name == "source_balanced_random":
        rng=random.Random(1); train=[]; test=[]; by=defaultdict(list)
        for r in records: by[str(r.get("candidate_source_type"))].append(r)
        for rows in by.values():
            rows=list(rows); rng.shuffle(rows)
            if len(rows)==1: train.extend(rows)
            else:
                cut=max(1,min(len(rows)-1,int(.7*len(rows)))); train.extend(rows[:cut]); test.extend(rows[cut:])
        return [(train,test,"source_balanced_random")] if test else split_records(records,"random")
    if name == "chronological":
        rows=sorted(records,key=lambda r:(rec_ts(r),str(r.get("candidate_id")))); cut=max(1,min(len(rows)-1,int(.7*len(rows)))); return [(rows[:cut], rows[cut:], "chronological")]
    if name == "leave_task_out":
        out=[]
        for t in sorted({r.get("task_id") for r in records}, key=lambda x: str(x)):
            tr=[r for r in records if r.get("task_id")!=t]; te=[r for r in records if r.get("task_id")==t]
            if tr and te: out.append((tr,te,f"task_{t}"))
        return out
    if name == "candidate_source_holdout":
        out=[]
        for src in sorted({r.get("candidate_source_type") for r in records}, key=lambda x: str(x)):
            tr=[r for r in records if r.get("candidate_source_type")!=src]; te=[r for r in records if r.get("candidate_source_type")==src]
            if len(tr)>=10 and len(te)>=3: out.append((tr,te,f"source_{src}"))
        return out
    raise ValueError(name)

def ymat(records, targets):
    y=np.zeros((len(records),len(targets)))
    for i,r in enumerate(records):
        m=metrics_of(r)
        for j,t in enumerate(targets): y[i,j]=m[t]
    return y



def _rank_array(a):
    a=np.asarray(a,float); order=np.argsort(a); ranks=np.empty(len(a),float); i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]: j+=1
        ranks[order[i:j]]=(i+j-1)/2.0; i=j
    return ranks

def spearman_corr(a,b):
    if len(a)<3: return None
    ra=_rank_array(a); rb=_rank_array(b); sa=float(np.std(ra)); sb=float(np.std(rb))
    if sa<=1e-12 or sb<=1e-12: return None
    return float(np.corrcoef(ra,rb)[0,1])

def met(y, p, train_y):
    if len(y)==0: return {"count":0,"mae":None,"normalized_mae":None,"r2":None,"range_coverage_accuracy":None}
    err=np.abs(p-y); mae=float(err.mean()); mean_abs=float(np.mean(np.abs(y))); nmae=mae/mean_abs if mean_abs>1e-12 else None
    ss_res=float(np.sum((y-p)**2)); ss_tot=float(np.sum((y-float(np.mean(y)))**2)); r2=None if len(y)<2 or ss_tot<=1e-12 else 1-ss_res/ss_tot
    rng=float(np.max(train_y)-np.min(train_y)) if len(train_y) else 0.0; acc=float(np.mean(err <= .2*rng)) if rng>1e-12 else None
    return {"count":int(len(y)),"mae":mae,"normalized_mae":nmae,"r2":r2,"spearman":spearman_corr(y,p),"range_coverage_accuracy":acc}


def pred_mean(tr, te, targets): return np.tile(ymat(tr,targets).mean(0),(len(te),1))

def pred_group(tr, te, targets, key):
    glob=ymat(tr,targets).mean(0); sums={}; cnt=Counter(); keys=key if isinstance(key, tuple) else (key,)
    for r in tr:
        k=tuple(r.get(x) for x in keys); sums[k]=sums.get(k,np.zeros(len(targets)))+ymat([r],targets)[0]; cnt[k]+=1
    means={k:sums[k]/cnt[k] for k in sums}; return np.vstack([means.get(tuple(r.get(x) for x in keys),glob) for r in te]) if te else np.zeros((0,len(targets)))

def pred_ridge(tr, te, targets, fs, alpha=1.0, include_task_id=True):
    vf=Vec().fit([features(r,fs,include_task_id=include_task_id) for r in tr]); x=vf.trans([features(r,fs,include_task_id=include_task_id) for r in tr]); xt=vf.trans([features(r,fs,include_task_id=include_task_id) for r in te]); y=ymat(tr,targets); log=bool(np.all(y>=0)); yf=np.log1p(y) if log else y
    xb=np.c_[np.ones(len(x)),x]; xtb=np.c_[np.ones(len(xt)),xt]; reg=np.eye(xb.shape[1])*alpha; reg[0,0]=0
    try: w=np.linalg.solve(xb.T@xb+reg, xb.T@yf)
    except np.linalg.LinAlgError: w=np.linalg.pinv(xb.T@xb+reg)@xb.T@yf
    p=xtb@w
    if log: p=np.maximum(np.expm1(p),0)
    return p


def pred_online(tr, te, targets, fs, lr=.01, epochs=2):
    vf=Vec().fit([features(r,fs) for r in tr]); x=vf.trans([features(r,fs) for r in tr]); xt=vf.trans([features(r,fs) for r in te]); y=ymat(tr,targets); log=bool(np.all(y>=0)); yf=np.log1p(y) if log else y
    w=np.zeros((x.shape[1]+1,len(targets))); xb=np.c_[np.ones(len(x)),x]
    for _ in range(epochs):
        for xi,yi in zip(xb,yf): w-=lr*np.outer(xi, xi@w-yi)
    preds=[]
    for row,r in zip(xt,te):
        xi=np.r_[1.0,row]; raw=xi@w; out=np.maximum(np.expm1(raw),0) if log else raw; preds.append(out)
        yi=ymat([r],targets)[0]; yi=np.log1p(yi) if log else yi; w-=lr*np.outer(xi, raw-yi)
    return np.vstack(preds) if preds else np.zeros((0,len(targets)))


def eval_split(tr, te, targets, split, detail, pred_rows):
    res={"train_count":len(tr),"test_count":len(te),"models":{}}; yt=ymat(te,targets); ytr=ymat(tr,targets)
    preds={"mean":pred_mean(tr,te,targets),"task_mean":pred_group(tr,te,targets,"task_id"),"source_mean":pred_group(tr,te,targets,"candidate_source_type"),"task_source_mean":pred_group(tr,te,targets,("task_id","candidate_source_type"))}
    for fs in FEATURE_SETS:
        preds[f"ridge_{fs}"]=pred_ridge(tr,te,targets,fs,include_task_id=True)
        preds[f"ridge_no_task_{fs}"]=pred_ridge(tr,te,targets,fs,include_task_id=False)
    ens=[preds[k] for k in ["ridge_F2","ridge_F4","ridge_F5"] if k in preds]
    if ens: preds["ensemble_ridge_F2_F4_F5"]=np.mean(np.stack(ens,axis=0),axis=0)
    if split=="chronological": preds["online_sgd_F4"]=pred_online(tr,te,targets,"F4")
    for name,p in preds.items():
        res["models"][name]={t:met(yt[:,j],p[:,j],ytr[:,j]) for j,t in enumerate(targets)}
        if name in {"ridge_F4","online_sgd_F4"}:
            for i,r in enumerate(te):
                for j,t in enumerate(targets): pred_rows.append({"split":split,"split_detail":detail,"model":name,"feature_set":"F4","record_index":r.get("_manifest_index"),"task_id":r.get("task_id"),"candidate_id":r.get("candidate_id"),"candidate_source_type":r.get("candidate_source_type"),"target_metric":t,"y_true":float(yt[i,j]),"y_pred":float(p[i,j]),"abs_error":float(abs(p[i,j]-yt[i,j]))})
    return res


def agg_leave(evals, targets):
    out={"heldout_task_count":len(evals),"models":{}}
    for model in sorted({m for e in evals for m in e["models"]}):
        out["models"][model]={}
        for t in targets:
            n=0; mae=0; nmae=0; nn=0; acc=0; na=0
            for e in evals:
                mm=e["models"].get(model,{}).get(t,{}) ; c=int(mm.get("count") or 0)
                if not c or mm.get("mae") is None: continue
                n+=c; mae+=mm["mae"]*c
                if mm.get("normalized_mae") is not None: nmae+=mm["normalized_mae"]*c; nn+=c
                if mm.get("range_coverage_accuracy") is not None: acc+=mm["range_coverage_accuracy"]*c; na+=c
            out["models"][model][t]={"count":n,"mae":mae/n if n else None,"normalized_mae":nmae/nn if nn else None,"r2":None,"range_coverage_accuracy":acc/na if na else None}
    return out

def predictability(splits, targets, tsum):
    out={}; rm=splits.get("random",{}).get("models",{}); cm=splits.get("chronological",{}).get("models",{}); lm=splits.get("leave_task_out",{}).get("models",{}); shm=splits.get("candidate_source_holdout",{}).get("models",{})
    for t in targets:
        best={}
        for label,models in [("random",rm),("chronological",cm),("leave_task_out",lm),("candidate_source_holdout",shm)]:
            b=None
            for name,pt in models.items():
                x=pt.get(t,{}); n=x.get("normalized_mae")
                if n is not None and (b is None or n<b["normalized_mae"]): b={"model":name,**x}
            best[label]=b
        base=rm.get("mean",{}).get(t,{}).get("normalized_mae"); imp=None
        if base and best["random"] and best["random"].get("normalized_mae") is not None: imp=(base-best["random"]["normalized_mae"])/base
        out[t]={"group":tsum[t]["group"],"random_mean_normalized_mae":base,"best_random":best["random"],"best_chronological":best["chronological"],"best_leave_task_out":best["leave_task_out"],"best_candidate_source_holdout":best.get("candidate_source_holdout"),"random_improvement_over_mean":imp}
    return out


def decide(metrics):
    tsum=metrics["target_metric_summary"]; main=[t for t in metrics["selected_targets"] if tsum[t]["group"]=="main"]; pred=metrics["target_predictability"]
    good=[]; chrono_good=[]; random_only=[]
    for t in main:
        p=pred[t]; imp=p.get("random_improvement_over_mean"); cn=(p.get("best_chronological") or {}).get("normalized_mae")
        cbase=metrics["splits"].get("chronological",{}).get("models",{}).get("mean",{}).get(t,{}).get("normalized_mae")
        cimp=None
        if cbase and cn is not None: cimp=(cbase-cn)/cbase
        p["chronological_improvement_over_mean"]=cimp
        if imp is not None and imp>0.10:
            good.append(t)
            if (cimp is not None and cimp>0.05) or (cn is not None and cn<0.80): chrono_good.append(t)
            else: random_only.append(t)
    def med_best(split):
        vals=[]
        for t in main:
            b=pred[t].get("best_"+split) if split in {"random","chronological"} else None
            if split=="leave_task_out": b=pred[t].get("best_leave_task_out")
            if split=="candidate_source_holdout": b=pred[t].get("best_candidate_source_holdout")
            n=(b or {}).get("normalized_mae")
            if n is not None: vals.append(float(n))
        return float(np.median(vals)) if vals else None
    unc=metrics.get("uncertainty_diagnostic",{})
    gate30=None
    for g in unc.get("simulated_gate",[]):
        if abs(float(g.get("profile_fraction",0))-0.30)<1e-9: gate30=g
    spearman=unc.get("overall_uncertainty_error_spearman")
    gate_ok=bool(gate30 and gate30.get("simulated_full_profile_call_reduction",0)>=0.30 and (gate30.get("relative_error_vs_predict_all") or 1.0)<=0.95)
    lto_med=med_best("leave_task_out"); src_med=med_best("candidate_source_holdout")
    lto_not_collapsed=bool(lto_med is not None and lto_med<2.0)
    uncertainty_catches=bool(spearman is not None and spearman>0.15 and gate_ok)
    if len(good)>=3 and len(chrono_good)>=3 and (lto_not_collapsed or uncertainty_catches) and gate_ok:
        status="ready_for_P07"; ready=True; needs=False; next_phase="P07"
    elif len(good)>=3 and len(chrono_good)>=1:
        status="needs_P05C_more_data"; ready=False; needs=True; next_phase="P05C"
    elif len(good)<3:
        status="rethink_targets"; ready=False; needs=True; next_phase="P05C"
    else:
        status="blocked_by_metric_coverage"; ready=False; needs=True; next_phase="P05C"
    return {"status":status,"next_phase":next_phase,"ready_for_p07":ready,"needs_p05c_more_data":needs,"good_main_metrics_random":good[:20],"good_main_metrics_chronological":chrono_good[:20],"random_only_metrics":random_only[:20],"leave_task_out_median_best_normalized_mae":lto_med,"candidate_source_holdout_median_best_normalized_mae":src_med,"uncertainty_error_spearman":spearman,"gate30":gate30,"ready_for_full_eval":False,"needs_p05b_more_data":False,"promising_main_metrics":good[:12],"leave_task_out_summary":"aggregated over held-out task ids","chronological_summary":"first 70% train, last 30% test by timestamp/order"}

def model_table(split, targets, main_only, tsum):
    rows=[]
    for m,pt in split.get("models",{}).items():
        vals=[]
        for t in targets:
            if main_only and tsum[t]["group"]!="main": continue
            n=pt.get(t,{}).get("normalized_mae")
            if n is not None and math.isfinite(float(n)): vals.append(float(n))
        rows.append({"model":m,"median_normalized_mae":float(np.median(vals)) if vals else None,"target_count":len(vals)})
    return sorted(rows,key=lambda r:(float("inf") if r["median_normalized_mae"] is None else r["median_normalized_mae"],r["model"]))



def uncertainty_diagnostic(records, targets):
    items=[]; per_target={}
    main_targets=[t for t in targets if any(t.startswith(p) for p in MAIN_PREFIXES)]
    for t in main_targets:
        rows=[r for r in records if t in metrics_of(r)]
        if len(rows)<20: continue
        tr,te,_=split_records(rows,"random")[0]
        y=ymat(te,[t])[:,0]; train_y=ymat(tr,[t])[:,0]
        preds=[]
        for fs in ["F2","F4","F5"]:
            preds.append(pred_ridge(tr,te,[t],fs)[:,0])
        mat=np.vstack(preds); ens=mat.mean(0); unc=mat.std(0); err=np.abs(ens-y)
        denom=float(np.mean(np.abs(y))) if float(np.mean(np.abs(y)))>1e-12 else 1.0
        per_target[t]={"count":len(te),"uncertainty_error_spearman":spearman_corr(unc,err),"ensemble_normalized_mae":float(np.mean(err)/denom)}
        for u,e in zip(unc,err): items.append({"target":t,"uncertainty":float(u),"normalized_abs_error":float(e/denom)})
    if not items: return {"status":"insufficient_uncertainty_samples","sample_count":0}
    unc=np.asarray([x["uncertainty"] for x in items],float); err=np.asarray([x["normalized_abs_error"] for x in items],float); order=np.argsort(unc)
    deciles=[]
    for d in range(10):
        lo=int(d*len(order)/10); hi=int((d+1)*len(order)/10); idx=order[lo:hi]
        if len(idx): deciles.append({"decile_low_to_high_uncertainty":d+1,"count":int(len(idx)),"mean_uncertainty":float(np.mean(unc[idx])),"mean_normalized_abs_error":float(np.mean(err[idx]))})
    base=float(np.mean(err)); high=np.argsort(-unc); gate=[]
    for frac in [0.10,0.20,0.30,0.40]:
        k=int(round(frac*len(items))); prof=set(high[:k]); residual=np.asarray([0.0 if i in prof else err[i] for i in range(len(items))],float)
        gate.append({"profile_fraction":frac,"simulated_full_profile_call_reduction":1.0-frac,"mean_normalized_abs_error_after_gate":float(np.mean(residual)),"relative_error_vs_predict_all":float(np.mean(residual)/base) if base>1e-12 else None})
    return {"status":"computed_offline_only_not_P07","sample_count":len(items),"target_count":len(per_target),"overall_uncertainty_error_spearman":spearman_corr(unc,err),"base_predict_all_mean_normalized_abs_error":base,"deciles":deciles,"simulated_gate":gate,"per_target":per_target}

def write_dataset(path,audit,tsum):
    lines=["# Phase 06A Dataset Audit","",f"- Generated: `{utc_timestamp()}`",f"- Input raw record count: `{audit['input_raw_record_count']}`",f"- Predictor-usable record count: `{audit['predictor_usable_record_count']}`",f"- Excluded record count: `{audit['excluded_record_count']}`",f"- Excluded reasons: `{audit['excluded_reasons']}`",f"- Complete-key record count: `{audit['complete_key_record_count']}`",f"- Chronological timestamp count: `{audit['chronological_timestamp_count']}`",f"- Leave-task-out meaningful: `{audit['leave_task_out_meaningful']}`",f"- Chronological split meaningful: `{audit['chronological_split_meaningful']}`","","## Task Distribution","","| Task | Count |","|---|---:|"]
    for k,v in audit["task_id_distribution"].items(): lines.append(f"| `{k}` | {v} |")
    lines += ["","## Candidate Source Distribution","","| Source type | Count |","|---|---:|"]
    for k,v in audit["candidate_source_type_distribution"].items(): lines.append(f"| `{k}` | {v} |")
    lines += ["","## Missing NCU Metric Summary","","| Metric | Coverage | Count |","|---|---:|---:|"]
    for k,it in tsum.items(): lines.append(f"| `{k}` | {it['coverage']:.3f} | {it['count']} |")
    path.write_text("\n".join(lines)+"\n")


def write_targets(path,tsum,selected):
    lines=["# Phase 06A Target Metric Selection","",f"- Generated: `{utc_timestamp()}`",""]
    for g in ["sanity","main","excluded"]:
        lines += [f"## {g.title()} Targets","","| Metric | Selected | Coverage | CV | Reason |","|---|---|---:|---:|---|"]
        for k,it in tsum.items():
            inc=(not it["selected"]) if g=="excluded" else (it["selected"] and it["group"]==g)
            if inc: lines.append(f"| `{k}` | `{it['selected']}` | {it['coverage']:.3f} | `{it['cv']}` | `{it['excluded_reason']}` |")
        lines.append("")
    lines += ["## Notes","","- `gpu__time_duration*` and `gpu__time_active*` are sanity targets because cheap latency can make them artificially easy.","- Main targets are memory traffic, warp activity, instruction count, and cycle metrics.","- Occupancy and stall metrics were not present in the selected NCU output for this dataset.",f"- Selected target count: `{len(selected)}`"]
    path.write_text("\n".join(lines)+"\n")


def write_report(path,metrics):
    tsum=metrics["target_metric_summary"]; targets=metrics["selected_targets"]; main=[t for t in targets if tsum[t]["group"]=="main"]; pred=metrics["target_predictability"]
    best=sorted([(t,p.get("random_improvement_over_mean"),(p.get("best_random") or {}).get("model"),(p.get("best_random") or {}).get("normalized_mae")) for t,p in pred.items() if t in main and p.get("random_improvement_over_mean") is not None], key=lambda x:(-(x[1] or -999),x[0]))
    worst=sorted([(t,p.get("random_improvement_over_mean"),(p.get("best_random") or {}).get("model"),(p.get("best_random") or {}).get("normalized_mae")) for t,p in pred.items() if t in main], key=lambda x:((x[1] if x[1] is not None else -999),x[0]))
    lines=["# Phase 06A Predictor Smoke","",f"- Generated: `{utc_timestamp()}`","- Scope: smoke/evaluability only, not a paper claim.",f"- Predictor-usable records: `{metrics['dataset_audit']['predictor_usable_record_count']}`",f"- Selected targets: `{len(targets)}` total, `{len(main)}` main targets.",f"- Decision: `{metrics['decision']['status']}`","","## Feature Set Ablation, Random Split, Main Targets","","| Model | Median normalized MAE | Target count |","|---|---:|---:|"]
    for r in model_table(metrics["splits"]["random"],targets,True,tsum): lines.append(f"| `{r['model']}` | `{r['median_normalized_mae']}` | {r['target_count']} |")
    lines += ["","## Feature Set Ablation, Chronological Split, Main Targets","","| Model | Median normalized MAE | Target count |","|---|---:|---:|"]
    for r in model_table(metrics["splits"]["chronological"],targets,True,tsum): lines.append(f"| `{r['model']}` | `{r['median_normalized_mae']}` | {r['target_count']} |")
    lines += ["","## Best Main Targets By Random Improvement","","| Metric | Improvement over mean | Best model | Best normalized MAE |","|---|---:|---|---:|"]
    for t,imp,m,n in best[:10]: lines.append(f"| `{t}` | `{imp}` | `{m}` | `{n}` |")
    lines += ["","## Worst Main Targets","","| Metric | Improvement over mean | Best model | Best normalized MAE |","|---|---:|---|---:|"]
    for t,imp,m,n in worst[:10]: lines.append(f"| `{t}` | `{imp}` | `{m}` | `{n}` |")
    lines += ["","## Distribution Shift","",f"- Leave-task-out summary: `{metrics['decision']['leave_task_out_summary']}`",f"- Chronological summary: `{metrics['decision']['chronological_summary']}`","","## Warning","","This smoke run is diagnostic only and must not be described as final CGO evidence."]
    path.write_text("\n".join(lines)+"\n")


def write_summary(path,metrics):
    d=metrics["decision"]; tsum=metrics["target_metric_summary"]; sel=metrics["selected_targets"]; main=[t for t in sel if tsum[t]["group"]=="main"]
    lines=["# Phase 06A Summary","",f"- Generated: `{utc_timestamp()}`",f"- Dataset size: `{metrics['dataset_audit']['input_raw_record_count']}` manifest records.",f"- Predictor-usable complete-key count: `{metrics['dataset_audit']['predictor_usable_record_count']}`",f"- Selected target metrics: `{len(sel)}` total, `{len(main)}` main.",f"- P06A decision: `{d['status']}`",f"- Ready for P06 full eval: `{d['ready_for_full_eval']}`",f"- Needs P05B more data: `{d['needs_p05b_more_data']}`","","## Selected Main Targets"]
    lines += [f"- `{t}`" for t in main]
    lines += ["","## Best Predictable Metrics"]
    lines += [f"- `{t}`" for t in d["promising_main_metrics"][:8]] or ["- none met the conservative smoke criterion across random and distribution-shift checks"]
    lines += ["","## Recommendation"]
    if d["needs_p05b_more_data"]: lines += ["- Recommended next phase: P05B targeted data expansion before making predictor claims.","- P06B may still run as a code-path smoke, but results should remain diagnostic only."]
    else: lines.append("- Recommended next phase: P06B full predictor evaluation.")
    path.write_text("\n".join(lines)+"\n")


def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return str(obj)

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default)+"\n")

def _median_table(metrics, split, group="main"):
    tsum=metrics["target_metric_summary"]; targets=[t for t in metrics["selected_targets"] if tsum[t]["group"]==group]; rows=[]
    models=sorted({m for t in targets for m in metrics["splits"].get(split,{}).get("models",{})})
    for m in models:
        vals=[]
        for t in targets:
            n=metrics["splits"].get(split,{}).get("models",{}).get(m,{}).get(t,{}).get("normalized_mae")
            if n is not None and math.isfinite(float(n)): vals.append(float(n))
        if vals: rows.append({"model":m,"target_count":len(vals),"median_normalized_mae":float(np.median(vals))})
    return sorted(rows,key=lambda r:(r["median_normalized_mae"],r["model"]))

def write_full_eval_reports(metrics, out_dir, report_path):
    out=Path(out_dir); tsum=metrics["target_metric_summary"]; decision=metrics["decision"]; pred=metrics["target_predictability"]
    write_json(out/"target_metric_groups.json", tsum)
    write_json(out/"split_metrics.json", metrics["splits"])
    write_json(out/"feature_ablation.json", {sp:{g:_median_table(metrics,sp,g) for g in ["sanity","main","v2_supplementary"]} for sp in ["random","source_balanced_random","chronological","leave_task_out","candidate_source_holdout"]})
    write_json(out/"uncertainty_diagnostic.json", metrics["uncertainty_diagnostic"])
    # Dataset audit
    a=metrics["dataset_audit"]; lines=["# Phase 06B Dataset Audit","",f"- Generated: `{utc_timestamp()}`"]
    for k in ["input_raw_record_count","predictor_usable_record_count","excluded_record_count","complete_key_record_count","complete_key_record_count_usable","manual_generated_count","real_llm_generated_count","task_count","candidate_source_type_count"]:
        lines.append(f"- {k}: `{a.get(k)}`")
    lines += [f"- Excluded reasons: `{a.get('excluded_reasons')}`",f"- Metric version distribution: `{a.get('metric_version_distribution')}`",f"- Missing field summary: `{a.get('missing_key_field_summary_all_records')}`",f"- Leave-task-out meaningful: `{a.get('leave_task_out_meaningful')}`",f"- Candidate-source holdout meaningful: `{a.get('candidate_source_holdout_meaningful')}`", "", "## Candidate Sources", "", "| Source | Count |", "|---|---:|"]
    for k,v in a.get("candidate_source_type_distribution",{}).items(): lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Tasks", "", "| Task | Count |", "|---|---:|"]
    for k,v in a.get("task_id_distribution",{}).items(): lines.append(f"| `{k}` | {v} |")
    Path("reports/phase_06b_dataset_audit.md").write_text("\n".join(lines)+"\n")
    # Targets
    lines=["# Phase 06B Target Metric Groups","",f"- Generated: `{utc_timestamp()}`","- Missing target metrics are not zero-filled; each metric is evaluated only on records where it exists.",""]
    for group,title in [("sanity","Group S: Sanity"),("main","Group M: Main"),("v2_supplementary","Group V2: Supplementary"),("excluded","Group X: Excluded")]:
        lines += [f"## {title}","","| Metric | Selected | Coverage | Count | Versions | Reason |","|---|---|---:|---:|---|---|"]
        for k,it in tsum.items():
            if it.get("group")==group: lines.append(f"| `{k}` | `{it['selected']}` | {it['coverage']:.3f} | {it['count']} | `{it.get('metric_profile_version_counts')}` | `{it.get('excluded_reason')}` |")
        lines.append("")
    Path("reports/phase_06b_target_metric_groups.md").write_text("\n".join(lines)+"\n")
    Path("reports/phase_06b_split_design.md").write_text("\n".join(["# Phase 06B Split Design","",f"- Generated: `{utc_timestamp()}`","","| Split | Purpose |","|---|---|","| `random` | Unique-candidate random split. |","| `source_balanced_random` | Candidate-source balanced random split. |","| `chronological` | Train on earlier records and test on later records. |","| `leave_task_out` | Hold out one KernelBench task at a time. |","| `candidate_source_holdout` | Hold out each candidate source type when sample count permits. |","","Raw `source_hash` and `candidate_config_hash` values are not used as categorical features."])+"\n")
    # Uncertainty
    u=metrics["uncertainty_diagnostic"]; lines=["# Phase 06B Uncertainty Diagnostic","",f"- Generated: `{utc_timestamp()}`","- Offline simulation only; P07 online profiling was not run.",f"- Status: `{u.get('status')}`",f"- Overall uncertainty/error Spearman: `{u.get('overall_uncertainty_error_spearman')}`", "", "## Deciles", "", "| Decile | Count | Mean uncertainty | Mean normalized abs error |", "|---:|---:|---:|---:|"]
    for d in u.get("deciles",[]): lines.append(f"| {d['decile_low_to_high_uncertainty']} | {d['count']} | {d['mean_uncertainty']} | {d['mean_normalized_abs_error']} |")
    lines += ["", "## Simulated Gate", "", "| Profile fraction | Full profile call reduction | Error after gate | Relative error |", "|---:|---:|---:|---:|"]
    for g in u.get("simulated_gate",[]): lines.append(f"| {g['profile_fraction']} | {g['simulated_full_profile_call_reduction']} | {g['mean_normalized_abs_error_after_gate']} | {g['relative_error_vs_predict_all']} |")
    Path("reports/phase_06b_uncertainty_diagnostic.md").write_text("\n".join(lines)+"\n")
    main=[t for t in metrics["selected_targets"] if tsum[t]["group"]=="main"]
    best=sorted([(t,pred[t].get("random_improvement_over_mean"),(pred[t].get("best_random") or {}).get("model"),(pred[t].get("best_random") or {}).get("normalized_mae"),(pred[t].get("best_chronological") or {}).get("normalized_mae"),(pred[t].get("best_leave_task_out") or {}).get("normalized_mae")) for t in main], key=lambda x:(-(x[1] if x[1] is not None else -999),x[0]))
    worst=sorted(best,key=lambda x:((x[1] if x[1] is not None else -999),x[0]))
    lines=["# Phase 06B Full Predictor Diagnostic Evaluation","",f"- Generated: `{utc_timestamp()}`","- Scope: diagnostic only, not final CGO paper evidence.",f"- Predictor-usable records: `{a.get('predictor_usable_record_count')}`",f"- Selected main targets: `{len(main)}`",f"- Decision: `{decision['status']}`", "", "## Feature-Set Ablation, Main Targets", ""]
    for sp in ["random","source_balanced_random","chronological","leave_task_out","candidate_source_holdout"]:
        lines += [f"### {sp}","","| Model | Median normalized MAE | Target count |","|---|---:|---:|"]
        for r in _median_table(metrics,sp,"main")[:14]: lines.append(f"| `{r['model']}` | `{r['median_normalized_mae']}` | {r['target_count']} |")
        lines.append("")
    lines += ["## Best Predictable Main Metrics","","| Metric | Random improvement | Best model | Random NMAE | Chrono NMAE | LTO NMAE |","|---|---:|---|---:|---:|---:|"]
    for row in best[:12]: lines.append(f"| `{row[0]}` | `{row[1]}` | `{row[2]}` | `{row[3]}` | `{row[4]}` | `{row[5]}` |")
    lines += ["", "## Worst / Unpredictable Main Metrics", "", "| Metric | Random improvement | Best model | Random NMAE | Chrono NMAE | LTO NMAE |", "|---|---:|---|---:|---:|---:|"]
    for row in worst[:12]: lines.append(f"| `{row[0]}` | `{row[1]}` | `{row[2]}` | `{row[3]}` | `{row[4]}` | `{row[5]}` |")
    lines += ["", "## Model Availability", "", "- RandomForestRegressor and HistGradientBoostingRegressor were not run because sklearn is not installed in the server venv."]
    Path(report_path).write_text("\n".join(lines)+"\n")
    summary=["# Phase 06B Summary","",f"- Generated: `{utc_timestamp()}`",f"- Dataset size: `{a.get('input_raw_record_count')}` manifest records.",f"- Predictor-usable count: `{a.get('predictor_usable_record_count')}`",f"- v1/v2 handling: metric targets are evaluated target-wise; v2-only stall/local metrics are supplementary and not zero-filled.",f"- P06B status: `{decision['status']}`",f"- Ready for P07: `{decision['ready_for_p07']}`",f"- Needs P05C more data: `{decision['needs_p05c_more_data']}`", "", "## Best Predictable Main Metrics"]
    summary += [f"- `{r[0]}`: random improvement `{r[1]}`, chronological NMAE `{r[4]}`, leave-task-out NMAE `{r[5]}`" for r in best[:10]]
    summary += ["", "## Worst / Unpredictable Main Metrics"]
    summary += [f"- `{r[0]}`: random improvement `{r[1]}`, chronological NMAE `{r[4]}`, leave-task-out NMAE `{r[5]}`" for r in worst[:10]]
    summary += ["", "## Uncertainty Diagnostic", f"- Overall uncertainty/error Spearman: `{u.get('overall_uncertainty_error_spearman')}`", f"- Simulated 30% profile gate: `{decision.get('gate30')}`", "", "## Claim Boundary", "- This can support a diagnostic systems claim, not yet a final CGO result claim; P07 must validate online gating."]
    Path("reports/phase_06b_summary.md").write_text("\n".join(summary)+"\n")
    next_cmd="python scripts/run_search_experiment.py --level 1 --task 1 --mode online_profbridge --mock --max-iters 1 --device 0" if decision["ready_for_p07"] else "python scripts/audit_unique_profile_pairs.py --input 'results/profile_pairs/*.jsonl' --out reports/phase_05c_precheck.md"
    Path("reports/phase_status.md").write_text("\n".join(["# Phase Status","", "- Current Phase: `P06B - Full Numerical ProfBridge Predictor Evaluation`", f"- Phase Status: `{decision['status']}`", "- Server-only: `true`", "- Local fetch: `forbidden`", f"- Updated: `{utc_timestamp()}`", "", "## P06B Result Summary", f"- Predictor-usable records: `{a.get('predictor_usable_record_count')}`", f"- Main metrics with random signal: `{len(decision['good_main_metrics_random'])}`", f"- Main metrics with chronological signal: `{len(decision['good_main_metrics_chronological'])}`", f"- Leave-task-out median best normalized MAE: `{decision.get('leave_task_out_median_best_normalized_mae')}`", f"- Candidate-source holdout median best normalized MAE: `{decision.get('candidate_source_holdout_median_best_normalized_mae')}`", f"- Ready for P07: `{decision['ready_for_p07']}`", f"- Needs P05C: `{decision['needs_p05c_more_data']}`", "", "## Next Recommended Command", "", "```bash", "cd /home/hansol/research/profbridge", "source .venv/bin/activate || true", next_cmd, "```"])+"\n")

def main():
    ap=argparse.ArgumentParser(description="Train/evaluate ProfBridge numerical predictor v0 smoke.")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/predictor/phase_06a")
    ap.add_argument("--report", default="reports/phase_06a_predictor_smoke.md")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full-eval", action="store_true", help="Run P06B full diagnostic evaluation reports.")
    args=ap.parse_args(); start=time.perf_counter(); out=ensure_dir(args.out_dir)
    paths=expand_inputs(args.input); records,audit=load_dataset(paths); tsum=target_summary(records); targets=sorted([k for k,v in tsum.items() if v["selected"]])
    if not records or not targets: raise SystemExit("no predictor-usable records or selected targets")
    pred_rows=[]; splits={}
    for s in ["random","source_balanced_random","chronological"]:
        tr,te,detail=split_records(records,s)[0]; splits[s]=eval_split(tr,te,targets,s,detail,pred_rows)
    leave=[]
    for tr,te,detail in split_records(records,"leave_task_out"): leave.append(eval_split(tr,te,targets,"leave_task_out",detail,pred_rows))
    splits["leave_task_out"]=agg_leave(leave,targets); splits["leave_task_out_details"]=leave
    source_hold=[]
    for tr,te,detail in split_records(records,"candidate_source_holdout"): source_hold.append(eval_split(tr,te,targets,"candidate_source_holdout",detail,pred_rows))
    splits["candidate_source_holdout"]=agg_leave(source_hold,targets); splits["candidate_source_holdout_details"]=source_hold
    metrics={"generated_at":utc_timestamp(),"scope":"smoke/evaluability only; not paper claim","git_commit":get_git_commit(REPO_ROOT),"input_files":[str(p) for p in paths],"dataset_audit":audit,"target_metric_summary":tsum,"selected_targets":targets,"sanity_targets":[t for t in targets if tsum[t]["group"]=="sanity"],"main_targets":[t for t in targets if tsum[t]["group"]=="main"],"excluded_targets":{k:v for k,v in tsum.items() if not v["selected"]},"feature_sets":{"F0":"task_id + candidate_source_type + generation_method","F1":"F0 + latency stats","F2":"F1 + static/source cheap features","F3":"F2 + ptxas/compiler/cost features when available","F4":"F3 + transformation history and candidate metadata, excluding raw identity hashes","F5":"F4 + metric-profile-version-aware feature"},"splits":splits,"uncertainty_diagnostic":uncertainty_diagnostic(records,targets),"unavailable_models":{"sklearn_random_forest":"sklearn not installed","sklearn_hist_gradient_boosting":"sklearn not installed"},"timing":{"total_train_eval_time_sec":time.perf_counter()-start}}
    metrics["target_predictability"]=predictability(splits,targets,tsum); metrics["decision"]=decide(metrics)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)); (out/"target_metric_summary.json").write_text(json.dumps(tsum,indent=2,sort_keys=True)); write_json(out/"target_metric_groups.json",tsum); write_json(out/"split_metrics.json",splits); write_json(out/"uncertainty_diagnostic.json",metrics["uncertainty_diagnostic"]); write_json(out/"feature_ablation.json", {sp:{g:_median_table(metrics,sp,g) for g in ["sanity","main","v2_supplementary"]} for sp in ["random","source_balanced_random","chronological","leave_task_out","candidate_source_holdout"]}); write_jsonl(out/"predictions.jsonl",pred_rows)
    write_dataset(Path("reports/phase_06a_dataset_audit.md"),audit,tsum); write_targets(Path("reports/phase_06a_target_metric_selection.md"),tsum,targets); write_report(Path(args.report),metrics); write_summary(Path("reports/phase_06a_summary.md"),metrics)
    if args.full_eval:
        write_full_eval_reports(metrics, out, Path(args.report))
    for p in [out/"metrics.json", out/"predictions.jsonl", out/"target_metric_summary.json", Path(args.report), Path("reports/phase_06a_dataset_audit.md"), Path("reports/phase_06a_target_metric_selection.md"), Path("reports/phase_06a_summary.md")]: print(p)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
