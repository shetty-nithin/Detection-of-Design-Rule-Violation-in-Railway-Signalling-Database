import os
import numpy as np, pandas as pd, torch
from sydb.loader import load_sydb
from sydb.graph import union_spec, spec_for, build_graph
from sydb.labels import read_label_table
 
 
def tensors(G):
    return dict(feats={t: torch.from_numpy(x) for t, x in G["feats"].items()},
                src=torch.from_numpy(G["src"]).long(), dst=torch.from_numpy(G["dst"]).long(),
                rel=torch.from_numpy(G["rel"]).long(), attr=torch.from_numpy(G["attr"]))
 
 
def discover(data_dir="data", projects_csv=None):
    """Projects either from a CSV (project_id,xml_path,label_path,group) or from folders
    data/<name>/sydb.xml + data/<name>/labels.xlsx."""
    out = []
    if projects_csv:
        df = pd.read_csv(projects_csv)
        for _, r in df.iterrows():
            out.append(dict(id=str(r["project_id"]), xml=r["xml_path"], labels=r["label_path"],
                            group=str(r.get("group", r["project_id"]))))
    else:
        for name in sorted(os.listdir(data_dir)):
            x, l = os.path.join(data_dir, name, "sydb.xml"), os.path.join(data_dir, name, "labels.xlsx")
            if os.path.exists(x) and os.path.exists(l):
                out.append(dict(id=name, xml=x, labels=l, group=name))
    if not out:
        raise SystemExit(f"no projects found in {data_dir} (need <name>/sydb.xml and <name>/labels.xlsx)")
    return out
 
 
class Project:
    pass
 
 
def load_all(projects, exclude_rules=(), verbose=True):
    raw = [load_sydb(p["xml"], verbose=False) for p in projects]
    master = union_spec(raw)
    tabs_l = [read_label_table(p["labels"]) for p in projects]
    rules = []
    for t in tabs_l:
        for r in t["rule"]:
            if r not in rules and r not in exclude_rules:
                rules.append(r)
    out = []
    for p, tb, lab in zip(projects, raw, tabs_l):
        sp, tabs, missing = spec_for(master, tb)
        G = build_graph(tabs, sp)
        pr = Project()
        pr.id, pr.group, pr.spec, pr.T = p["id"], p["group"], sp, tensors(G)
        pr.y = np.full(len(rules), -1, dtype=np.int64)
        d = dict(zip(lab["rule"], lab["y"]))
        for j, r in enumerate(rules):
            pr.y[j] = d.get(r, -1)
        pr.labels = lab
        pr.tables = tabs
        out.append(pr)
        if verbose:
            print(f"[data] {p['id']:10s} objects={sp.n_nodes:6d} edges={len(G['src']):6d} "
                  f"labelled rules={int((pr.y >= 0).sum()):2d} NOK={int((pr.y == 1).sum()):2d} "
                  f"absent sections={len([m for m in missing if '.' not in m])}")
    return master, out, rules
