import argparse, pickle, numpy as np, pandas as pd, torch
from sydb.loader import load_sydb
from sydb.graph import conform, build_graph
from sydb.model import RuleGNN
from sydb.data import tensors
 
 
def tier(s, hi=0.5, lo=0.1):
    return "SUSPECT" if s >= hi else ("NEEDS REVIEW" if s >= lo else "likely OK")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sydb", required=True); ap.add_argument("--model", default="runs/demo")
    ap.add_argument("--out", default="suspects.xlsx"); ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()
    ck = torch.load(f"{a.model}/model.pt", weights_only=False)
    spec = pickle.load(open(f"{a.model}/spec.pkl", "rb"))
    raw = load_sydb(a.sydb, verbose=False)
    tabs, missing = conform(raw, spec)
    spec.rebind(tabs)
    if missing:
        print(f"[warn] {len(missing)} sheets/columns of the trained schema are absent (e.g. {missing[:3]}); filled as empty")
    G = build_graph(tabs, spec); T = tensors(G)
    model = RuleGNN({t: spec.dims[t] for t in spec.tables}, spec.n_rel, len(ck["rules"]), d=ck["d"])
    model.load_state_dict(ck["state"]); model.eval()
    with torch.no_grad():
        h = model.embed(T["feats"], spec.offset, spec.n_nodes, T["src"], T["dst"], T["rel"], T["attr"])
        S = torch.sigmoid(model(h)).numpy()
    owner = np.concatenate([[t] * len(tabs[t]) for t in spec.tables])
    local = np.concatenate([np.arange(len(tabs[t])) for t in spec.tables])
    rows, summ = [], []
    for j, r in enumerate(ck["rules"]):
        order = np.argsort(-S[:, j])
        summ.append(dict(rule=r, max_score=float(S[order[0], j]), tier=tier(S[order[0], j]),
                         n_objects_above_0_5=int((S[:, j] >= 0.5).sum())))
        for rank, i in enumerate(order[:a.top], 1):
            d = tabs[owner[i]]; nm = d[".Name"].iloc[local[i]] if ".Name" in d.columns else local[i]
            rows.append(dict(rule=r, rank=rank, sheet=owner[i], object=nm, score=float(S[i, j]), tier=tier(S[i, j])))
    with pd.ExcelWriter(a.out) as w:
        pd.DataFrame(summ).to_excel(w, "Summary", index=False)
        pd.DataFrame(rows).to_excel(w, "Suspects", index=False)
    print(pd.DataFrame(summ).to_string(index=False)); print("written", a.out)
 
 
if __name__ == "__main__":
    main()
