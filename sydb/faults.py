import copy
import numpy as np
import pandas as pd
from sydb.loader import is_ref_null, split_list, to_num
from .graph import KP_BEGIN, KP_END
 
RULES = ["F1_VALUE_SPIKE", "F2_KP_ORDER_SWAPPED", "F3_OFF_TRACK", "F4_WRONG_REFERENCE",
         "F5_DANGLING_REFERENCE", "F6_RARE_CATEGORY", "F7_LENGTH_MISMATCH"]
 
 
def _num_cols(spec, t):
    return [c for c in spec.cols[t]["num"] if not c.startswith("Kp")]
 
 
def inject(tables, spec, rule, rng, allowed_tables, k=1):
    """Inject k faults of family `rule` into tables (modified in place).
    Returns list of global node indices that are the offenders."""
    out = []
    for _ in range(k):
        cand = [t for t in allowed_tables if t in spec.tables]
        rng.shuffle(cand)
        for t in cand:
            d, r = tables[t], spec.cols[t]
            if len(d) == 0:
                continue
            i = int(rng.integers(len(d)))
            if rule == "F1_VALUE_SPIKE":
                cols = _num_cols(spec, t)
                if not cols:
                    continue
                c = cols[int(rng.integers(len(cols)))]
                v = to_num(d.at[i, c])
                if v is None or v == 0:
                    continue
                d.at[i, c] = v * float(rng.choice([10, 50, -5]))
            elif rule == "F2_KP_ORDER_SWAPPED":
                if KP_BEGIN not in d.columns or KP_END not in d.columns:
                    continue
                b, e = to_num(d.at[i, KP_BEGIN]), to_num(d.at[i, KP_END])
                if b is None or e is None or b == e:
                    continue
                d.at[i, KP_BEGIN], d.at[i, KP_END] = e, b
            elif rule == "F3_OFF_TRACK":
                pc = "Kp.Value" if "Kp.Value" in d.columns else None
                if pc is None or "Track_ID" not in d.columns:
                    continue
                v = to_num(d.at[i, pc])
                if v is None:
                    continue
                d.at[i, pc] = v + float(rng.choice([-1, 1])) * float(rng.integers(300_000, 2_000_000))
            elif rule in ("F4_WRONG_REFERENCE", "F5_DANGLING_REFERENCE"):
                rc = [c for c in r["ref"] if (t, c) in spec.rel_target and not is_ref_null(d.at[i, c])]
                if not rc:
                    continue
                c = rc[int(rng.integers(len(rc)))]
                toks = split_list(d.at[i, c])
                if rule == "F5_DANGLING_REFERENCE":
                    toks[0] = "ZZ_UNKNOWN_" + str(int(rng.integers(1000)))
                else:
                    tgt = spec.rel_target[(t, c)]
                    pool = [n for n in spec.names[tgt] if n not in toks]
                    if not pool:
                        continue
                    toks[0] = pool[int(rng.integers(len(pool)))]
                d.at[i, c] = ";".join(toks)
            elif rule == "F6_RARE_CATEGORY":
                if not r["cat"]:
                    continue
                c = r["cat"][int(rng.integers(len(r["cat"])))]
                vc = d[c].dropna().astype(str).value_counts()
                if len(vc) < 2:
                    continue
                d.at[i, c] = vc.index[-1]            # rarest value of that column
                if str(d.at[i, c]) == str(vc.index[0]):
                    continue
            elif rule == "F7_LENGTH_MISMATCH":
                lc = [c for c in d.columns if c.endswith("Length")]
                if not lc:
                    continue
                v = to_num(d.at[i, lc[0]])
                if v is None:
                    continue
                d.at[i, lc[0]] = v + float(rng.choice([-1, 1])) * float(rng.integers(200, 3000))
            else:
                raise ValueError(rule)
            out.append(spec.offset[t] + i)
            break
    return out
 
 
def make_project(base_tables, spec, rng, allowed_tables, p_nok=0.5, kmax=3):
    """One synthetic project: corrupted copy + per-rule labels + offender lists."""
    tables = {t: d.astype(object) for t, d in base_tables.items()}
    labels, offenders = {}, {}
    for rule in RULES:
        if rng.random() < p_nok:
            off = inject(tables, spec, rule, rng, allowed_tables, k=int(rng.integers(1, kmax + 1)))
        else:
            off = []
        labels[rule] = int(len(off) > 0)
        offenders[rule] = off
    return tables, labels, offenders
