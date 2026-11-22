"""
train.py - end-to-end DEMO: build graphs, make labelled synthetic projects, train the
rule-conditioned GNN, compare with baselines, save checkpoint + results.

    python train.py --sydb MML4-SYS-0113_A.xlsm --out runs/demo
"""
import argparse, json, os, pickle, time
import numpy as np, torch, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sydb.loader import load_sydb
from sydb.graph import Spec, build_graph
from sydb.faults import RULES, make_project
from sydb.model import RuleGNN


def tensors(G, spec):
    return dict(feats={t: torch.from_numpy(x) for t, x in G["feats"].items()},
                src=torch.from_numpy(G["src"]).long(), dst=torch.from_numpy(G["dst"]).long(),
                rel=torch.from_numpy(G["rel"]).long(), attr=torch.from_numpy(G["attr"]))


def summary_feats(G, spec):
    """Hand-crafted per-object features for the non-graph baselines (same raw information,
    no neighbours)."""
    N = spec.n_nodes
    X = np.zeros((N, 11), np.float32)
    for k, t in enumerate(spec.tables):
        r = spec.cols[t]
        f = G["feats"][t]
        o = spec.offset[t]
        n = f.shape[0]
        i = 0
        nz = f[:, i:i + 2 * len(r["num"]):2]; i += 2 * len(r["num"])
        i += 2 * len(r["bool"])
        cr = f[:, i:i + 2 * len(r["cat"]):2]; i += 2 * len(r["cat"])
        rf = f[:, i:i + 2 * len(r["ref"])]; i += 2 * len(r["ref"])
        der = f[:, -5:]
        blk = X[o:o + n]
        if nz.shape[1]: blk[:, 0] = np.abs(nz).max(1)
        if cr.shape[1]: blk[:, 1] = cr.max(1)
        if rf.shape[1]: blk[:, 2] = (rf[:, 0::2] - rf[:, 1::2]).sum(1)
        blk[:, 3:8] = der
        blk[:, 8] = k
    R = len(spec.rels)
    for rel_id in (2 * R, 2 * R + 1):
        m = G["rel"] == rel_id
        X[G["dst"][m], 9 if rel_id == 2 * R else 10] = np.abs(G["attr"][m])
    return X


def outlier_score(X):
    return X[:, 0] / 10 + X[:, 1] / 3 + np.clip(X[:, 2], 0, None) + (X[:, 3] < 0) + X[:, 6] + X[:, 7] + X[:, 5]


class Proj:
    pass


def gen(base, spec, n, seed, tabs):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        tb, lab, off = make_project(base, spec, rng, tabs)
        G = build_graph(tb, spec)
        p = Proj()
        p.G, p.T = G, tensors(G, spec)
        p.X = summary_feats(G, spec)
        p.y = np.array([lab[r] for r in RULES])
        p.Y = np.zeros((spec.n_nodes, len(RULES)), np.float32)
        for j, r in enumerate(RULES):
            p.Y[off[r], j] = 1
        out.append(p)
    return out


def node_mask(spec, tabs):
    m = np.zeros(spec.n_nodes, bool)
    for t in tabs:
        m[spec.offset[t]:spec.offset[t] + int(spec.n_by_table[t])] = True
    return m


def evaluate(name, score_fn, projs, mask, k=10):
    rows = {}
    for j, r in enumerate(RULES):
        ys, ss, hit, rl_y, rl_s = [], [], [], [], []
        for p in projs:
            s = score_fn(p, j)[mask]
            y = p.Y[mask, j]
            ys.append(y); ss.append(s)
            rl_y.append(p.y[j]); rl_s.append(s.max())
            if y.sum() > 0:
                hit.append(float(y[np.argsort(-s)[:k]].sum() > 0))
        ys, ss = np.concatenate(ys), np.concatenate(ss)
        rows[r] = dict(auprc=float(average_precision_score(ys, ss)) if ys.sum() else None,
                       top10_hit=float(np.mean(hit)) if hit else None,
                       rule_auroc=float(roc_auc_score(rl_y, rl_s)) if 0 < sum(rl_y) < len(rl_y) else None,
                       n_pos=int(ys.sum()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sydb", required=True)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_test", type=int, default=14)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    base = load_sydb(a.sydb, verbose=False)
    spec = Spec(base)
    spec.n_by_table = {t: len(base[t]) for t in spec.tables}
    elig = [t for t in spec.tables if len(base[t]) >= 5 and t not in ("Projects_Cap", "Urbalis_Sectors_Cap")]
    rng = np.random.default_rng(a.seed)
    held = list(rng.choice(elig, size=max(3, len(elig) // 4), replace=False))
    seen = [t for t in elig if t not in held]
    print(f"[data] {spec.n_nodes} objects, {len(spec.rels)} reference types; "
          f"{len(seen)} seen tables, {len(held)} held-out tables")
    train = gen(base, spec, a.n_train, 100 + a.seed, seen)
    test_seen = gen(base, spec, a.n_test, 200 + a.seed, seen)
    test_held = gen(base, spec, a.n_test, 300 + a.seed, held)
    print(f"[data] {len(train)}+{len(test_seen)}+{len(test_held)} synthetic projects in {time.time()-t0:.0f}s")
    m_seen, m_held = node_mask(spec, seen), node_mask(spec, held)

    # ------------------------------------------------------------ GNN
    model = RuleGNN({t: spec.dims[t] for t in spec.tables}, spec.n_rel, len(RULES), d=a.d)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    off_t = spec.offset
    mt = torch.from_numpy(m_seen)

    def fwd(p):
        h = model.embed(p.T["feats"], off_t, spec.n_nodes, p.T["src"], p.T["dst"], p.T["rel"], p.T["attr"])
        return model(h)

    for ep in range(a.epochs):
        model.train(); tot = 0
        for ix in np.random.permutation(len(train)):
            p = train[ix]
            logit = fwd(p)
            Y = torch.from_numpy(p.Y)
            lo = logit[mt]; yo = Y[mt]
            obj = F.binary_cross_entropy_with_logits(lo, yo, pos_weight=torch.tensor(50.0))
            pooled = lo.max(0).values                                   # multiple-instance pooling
            rule = F.binary_cross_entropy_with_logits(pooled, torch.from_numpy(p.y).float())
            loss = rule + 5 * obj
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item()
        print(f"[train] epoch {ep+1}/{a.epochs} loss {tot/len(train):.3f}  ({time.time()-t0:.0f}s)")
    model.eval()
    cache = {}

    def gnn_score(p, j):
        if id(p) not in cache:
            with torch.no_grad():
                cache[id(p)] = torch.sigmoid(fwd(p)).numpy()
        return cache[id(p)][:, j]

    # ------------------------------------------------------------ baselines
    gbts = []
    for j in range(len(RULES)):
        Xs = np.concatenate([p.X[m_seen] for p in train]); ys = np.concatenate([p.Y[m_seen, j] for p in train])
        gbts.append(HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, class_weight="balanced", early_stopping=False).fit(Xs, ys)
                    if ys.sum() else None)
    gcache = {}

    def gbt_score(p, j):
        if (id(p), j) not in gcache:
            gcache[(id(p), j)] = gbts[j].predict_proba(p.X)[:, 1] if gbts[j] else np.zeros(spec.n_nodes)
        return gcache[(id(p), j)]

    def out_score(p, j):
        return outlier_score(p.X)

    def rng_score(p, j):
        return np.random.default_rng(1).random(spec.n_nodes)

    res = {}
    for split, projs, mask in (("seen_tables_new_projects", test_seen, m_seen), ("HELD_OUT_tables", test_held, m_held)):
        res[split] = {nm: evaluate(nm, fn, projs, mask) for nm, fn in
                      (("random", rng_score), ("outlier_score", out_score), ("gradient_boosting", gbt_score),
                       ("rule_conditioned_GNN", gnn_score))}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    torch.save(dict(state=model.state_dict(), d=a.d, rules=RULES), os.path.join(a.out, "model.pt"))
    pickle.dump(spec, open(os.path.join(a.out, "spec.pkl"), "wb"))

    # ------------------------------------------------------------ report
    def avg(split, nm, key):
        v = [x[key] for x in res[split][nm].values() if x[key] is not None]
        return float(np.mean(v)) if v else float("nan")
    lines = ["DEMO RESULTS (synthetic fault injection into ONE real SyDB; see README for what this does NOT prove)",
             f"objects={spec.n_nodes}  train projects={len(train)}  test projects per split={a.n_test}", ""]
    for split in res:
        lines += [f"== {split} ==", f"{'method':24s} {'mean AUPRC':>11s} {'top-10 hit':>11s} {'rule AUROC':>11s}"]
        for nm in res[split]:
            lines.append(f"{nm:24s} {avg(split,nm,'auprc'):11.3f} {avg(split,nm,'top10_hit'):11.3f} {avg(split,nm,'rule_auroc'):11.3f}")
        lines.append("per-rule AUPRC  (random | outlier | GBT | GNN):")
        for r in RULES:
            lines.append("  %-24s " % r + " | ".join("%s" % ("  n/a" if res[split][nm][r]["auprc"] is None else "%5.2f" % res[split][nm][r]["auprc"])
                         for nm in res[split]) + f"   (positives={res[split]['random'][r]['n_pos']})")
        lines.append("")
    open(os.path.join(a.out, "results.txt"), "w").write("\n".join(lines))
    print("\n".join(lines)); print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
