import re
import numpy as np
import pandas as pd
from collections import defaultdict
from sydb.loader import is_ref_null, split_list, to_num, to_bool, is_null
 
KP_POINT = re.compile(r"^Kp\.Value$")
KP_BEGIN, KP_END = "Kp_Begin.Value", "Kp_End.Value"
REF_COL = re.compile(r"(_ID$|_ID_List$|_List$|_IDs$)", re.I)
SKIP_COL = re.compile(r"(Corrected_Trolley|Corrected_Gap|Object Reference|^Object|Adjustment|^DM_|\.Measurable|"
                      r"\.Position_Locked|Comment|Description|Remark)", re.I)
 
 
# ----------------------------------------------------------------------------------------
def _name_score(stem, section):
    """How well does a column stem (SSP, Track, Next_Up_Block, Remote_IO ...) match a section name?
    5 = stem equals the section (ignoring plural s and acronyms), 4 = a trailing part of the stem
    does (Real_Install_Position_Track -> Tracks), 2 = loose containment, 0 = no match."""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower()).replace("s", "")
    words = [w for w in section.split("_") if w]
    aliases = {norm(section)}
    for i in range(2, len(words) + 1):                       # Service_Stopping_Points -> SSP
        aliases.add(norm("".join(w[0] for w in words[:i]) + "".join(words[i:])))
    sw = [w for w in stem.split("_") if w]
    if norm(stem) in aliases:
        return 5
    for i in range(1, len(sw)):
        if norm("_".join(sw[i:])) in aliases:
            return 4
    a, b = norm(stem), norm(section)
    if len(a) >= 3 and (b.startswith(a) or a in b):
        return 2
    return 0
 
 
class Spec:
    """Structure that is fixed for a schema (computed once from a reference project)."""
 
    def __init__(self, tables):
        self.tables = [t for t, d in tables.items() if len(d) > 0]
        self.rebind(tables)
        self._learn(tables)
 
    def rebind(self, tables):
        """(Re)compute node offsets and name lookup for a project with the same schema."""
        self.offset, n = {}, 0
        for t in self.tables:
            self.offset[t] = n
            n += len(tables[t]) if t in tables else 0
        self.n_nodes = n
        self.names = {}
        for t in self.tables:
            d = tables[t]
            nm = d[".ID"] if ".ID" in d.columns else []          # references use the numeric ID
            self.names[t] = {str(v): self.offset[t] + i for i, v in enumerate(nm) if v is not None}
        self.node_table = np.zeros(n, dtype=np.int64)
        for k, t in enumerate(self.tables):
            self.node_table[self.offset[t]:self.offset[t] + len(tables[t])] = k
        return self
 
    def _learn(self, tables):
        # column roles
        self.cols = {}
        for t in self.tables:
            d = tables[t]
            roles = {"num": [], "bool": [], "cat": [], "ref": []}
            for c in d.columns:
                if c in (".ID", ".Name") or SKIP_COL.search(c):
                    continue
                s = d[c]
                nn = s.dropna()
                nn = nn[[not is_null(v) for v in nn]]
                if len(nn) == 0:
                    continue
                if REF_COL.search(c):
                    roles["ref"].append(c)
                    continue
                vals = set(str(v).strip().lower() for v in nn.unique())
                if vals <= {"true", "false", "1", "0"}:
                    roles["bool"].append(c)
                    continue
                num = [to_num(v) for v in nn]
                if sum(x is not None for x in num) >= 0.95 * len(nn):
                    if nn.nunique() > 1:
                        roles["num"].append(c)
                    continue
                if nn.nunique() <= 40:
                    roles["cat"].append(c)
            self.cols[t] = roles
        # reference relation inference: a column X_ID holds IDs of objects of some section.
        # Candidates = sections whose ID set contains >= 90 % of the values; the column name
        # (e.g. SSP_ID -> Service_Stopping_Points via initials) decides between candidates.
        # If it stays ambiguous the column is NOT turned into edges (better no edge than a wrong one).
        self.rel_target = {}
        for t in self.tables:
            for c in self.cols[t]["ref"]:
                toks = []
                for v in tables[t][c]:
                    if not is_ref_null(v):
                        toks += split_list(v)
                if not toks:
                    continue
                stem = re.sub(r"(_ID_List|_IDs|_ID|_List)$", "", c.split("/")[-1], flags=re.I)
                cands = []
                for tt in self.tables:
                    ids = self.names[tt]
                    cov = sum(x in ids for x in toks) / len(toks)
                    if cov >= 0.9:
                        cands.append((_name_score(stem, tt), cov, tt))
                if not cands:
                    continue
                cands.sort(reverse=True)
                if cands[0][0] > 0 or len(cands) == 1:
                    self.rel_target[(t, c)] = cands[0][2]
        self.rels = {rc: i for i, rc in enumerate(sorted(self.rel_target))}
        # relation ids: 0..R-1 forward refs, R..2R-1 reverse, 2R next, 2R+1 prev
        self.n_rel = 2 * len(self.rels) + 2
        self.dims = {}
        for t in self.tables:
            r = self.cols[t]
            self.dims[t] = 2 * len(r["num"]) + 2 * len(r["bool"]) + 2 * len(r["cat"]) + 2 * len(r["ref"]) + 5
 
 
# ----------------------------------------------------------------------------------------
def _kp_cols(d):
    if KP_BEGIN in d.columns and KP_END in d.columns:
        return KP_BEGIN, KP_END
    return None
 
 
def _num_series(s):
    return pd.Series([to_num(v) for v in s], dtype="float64")
 
 
def build_graph(tables, spec):
    """Return dict of numpy arrays: feats{table}, edge_src, edge_dst, edge_rel, edge_attr."""
    feats, nodepos = {}, {}
    # track extents for "inside track" feature
    tr_ext = {}
    if "Tracks" in tables and len(tables["Tracks"]):
        tc = tables["Tracks"]
        b, e = _num_series(tc[KP_BEGIN]), _num_series(tc[KP_END])
        for nm, bb, ee in zip(tc[".ID"], b, e):
            if not (np.isnan(bb) or np.isnan(ee)):
                tr_ext[str(nm)] = (min(bb, ee), max(bb, ee))
    for t in spec.tables:
        d, r = tables[t], spec.cols[t]
        n = len(d)
        cols = []
        for c in r["num"]:
            x = _num_series(d[c]).to_numpy()
            miss = np.isnan(x)
            if (~miss).sum() > 0:
                med = np.nanmedian(x)
                mad = np.nanmedian(np.abs(x[~miss] - med))
                sc = 1.4826 * mad if mad > 0 else (np.nanstd(x) or 1.0)
                z = np.clip((x - med) / (sc if sc > 0 else 1.0), -10, 10)
            else:
                z = np.zeros(n)
            cols += [np.nan_to_num(z), miss.astype(float)]
        for c in r["bool"]:
            v = [to_bool(q) for q in d[c]]
            cols += [np.array([1.0 if q else 0.0 for q in v]), np.array([q is None for q in v], float)]
        for c in r["cat"]:
            s = d[c].astype(object).where(d[c].notna(), None)
            cnt = s.value_counts()
            tot = max(int(cnt.sum()), 1)
            rar = np.array([(-np.log(cnt[q] / tot) if q is not None else 0.0) for q in s])
            cols += [rar / 3.0, np.array([q is None for q in s], float)]
        for c in r["ref"]:
            tgt = spec.rel_target.get((t, c))
            pres, resv = np.zeros(n), np.zeros(n)
            for i, v in enumerate(d[c]):
                if is_ref_null(v):
                    continue
                toks = split_list(v)
                if not toks:
                    continue
                pres[i] = 1.0
                if tgt is not None:
                    resv[i] = np.mean([tok in spec.names[tgt] for tok in toks])
                else:
                    resv[i] = 1.0
            cols += [pres, resv]
        # derived geometric features (5)
        der = np.zeros((n, 5))
        kc = _kp_cols(d)
        if kc:
            b, e = _num_series(d[kc[0]]).to_numpy(), _num_series(d[kc[1]]).to_numpy()
            ln = e - b
            der[:, 0] = np.nan_to_num(np.sign(ln))                       # +1 ok, -1 reversed
            der[:, 1] = np.nan_to_num(np.log1p(np.abs(ln)) / 10.0)       # length
            lcols = [c for c in d.columns if re.search(r"Length$", c)]
            if lcols:
                L = _num_series(d[lcols[0]]).to_numpy()
                der[:, 2] = np.nan_to_num(np.log1p(np.abs(np.abs(ln) - L)) / 10.0)  # length mismatch
        pc = None
        if "Kp.Value" in d.columns:
            pc = "Kp.Value"
        elif kc:
            pc = kc[0]
        if pc and "Track_ID" in d.columns:
            p = _num_series(d[pc]).to_numpy()
            nodepos[t] = (p, d["Track_ID"].astype(object).to_numpy())
            for i, (pp, tk) in enumerate(zip(p, d["Track_ID"])):
                ex = tr_ext.get(str(tk))
                if ex is None or np.isnan(pp):
                    continue
                span = max(ex[1] - ex[0], 1.0)
                der[i, 3] = np.clip((pp - ex[0]) / span, -2, 3) - 0.5
                der[i, 4] = float(pp < ex[0] or pp > ex[1])               # outside own track
        cols = np.column_stack(cols + [der]) if cols else der
        feats[t] = cols.astype(np.float32)
 
    # ---------------- edges
    src, dst, rel, attr = [], [], [], []
    R = len(spec.rels)
    for (t, c), rid in spec.rels.items():
        tgt = spec.rel_target[(t, c)]
        d = tables[t]
        base = spec.offset[t]
        for i, v in enumerate(d[c]):
            if is_ref_null(v):
                continue
            for tok in split_list(v):
                j = spec.names[tgt].get(tok)
                if j is not None:
                    src.append(base + i); dst.append(j); rel.append(rid); attr.append(0.0)
                    src.append(j); dst.append(base + i); rel.append(R + rid); attr.append(0.0)
    # geometry: sort every positioned object by (track, kp); link neighbours
    items = []
    for t, (p, tk) in nodepos.items():
        for i in range(len(p)):
            if not np.isnan(p[i]) and tk[i] is not None:
                items.append((str(tk[i]), p[i], spec.offset[t] + i))
    items.sort()
    for a, b in zip(items[:-1], items[1:]):
        if a[0] != b[0]:
            continue
        dist = b[1] - a[1]
        f = float(np.sign(dist) * np.log1p(abs(dist)) / 10.0)
        src.append(a[2]); dst.append(b[2]); rel.append(2 * R); attr.append(f)          # a -> b : next
        src.append(b[2]); dst.append(a[2]); rel.append(2 * R + 1); attr.append(-f)      # b -> a : prev
    return dict(feats=feats, src=np.array(src), dst=np.array(dst), rel=np.array(rel),
                attr=np.array(attr, dtype=np.float32))
 
 
def conform(tables, spec):
    """Make a new project's tables match the trained schema (missing sheets/columns -> empty)."""
    out, missing = {}, []
    for t in spec.tables:
        d = tables.get(t)
        need = {c for r in spec.cols[t].values() for c in r}
        if d is None:
            d = pd.DataFrame({c: [] for c in sorted(need | {".ID", ".Name"})})
            missing.append(t)
        else:
            add = [c for c in need if c not in d.columns]
            if add:
                d = pd.concat([d, pd.DataFrame({c: [None] * len(d) for c in add}, index=d.index)], axis=1)
                missing += [f"{t}.{c}" for c in add]
        out[t] = d.reset_index(drop=True)
    return out, missing
 
 
def union_spec(list_of_tables):
    """Learn ONE schema (column roles, reference targets) from several projects, so every project
    gets the same feature layout. Returns the master Spec; use spec_for() per project."""
    secs = {}
    for tb in list_of_tables:
        for t, d in tb.items():
            secs.setdefault(t, []).append(d)
    union = {t: pd.concat(ds, ignore_index=True) for t, ds in secs.items()}
    return Spec(union)
 
 
def spec_for(master, tables):
    """Project-specific copy of the master Spec (own node numbering and ID lookup) plus the
    project's tables conformed to the master schema."""
    import copy
    tabs, missing = conform(tables, master)
    sp = copy.copy(master)
    sp.rebind(tabs)
    return sp, tabs, missing
