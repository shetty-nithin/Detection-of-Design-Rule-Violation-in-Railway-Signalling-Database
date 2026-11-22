import argparse, json, os, pickle, time
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sydb.data import discover, load_all
from sydb.model import RuleGNN
 
 
def fit(P, train_idx, rules, master, seed, epochs, d, lr, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RuleGNN({t: master.dims[t] for t in master.tables}, master.n_rel, len(rules), d=d, dropout=0.2)
    model.head[-1].bias.data.fill_(-3.0)                   # start by predicting "OK" everywhere
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    for ep in range(epochs):
        model.train(); tot = 0.0; n = 0
        for i in np.random.permutation(train_idx):
            p = P[i]
            m = torch.from_numpy(p.y >= 0)
            if m.sum() == 0:
                continue
            h = model.embed(p.T["feats"], p.spec.offset, p.spec.n_nodes, p.T["src"], p.T["dst"], p.T["rel"], p.T["attr"])
            pooled = model(h).max(0).values                  # multiple-instance pooling over objects
            loss = F.binary_cross_entropy_with_logits(pooled[m], torch.from_numpy(p.y).float()[m])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item(); n += 1
        if log and (ep + 1) % 10 == 0:
            log(f"      epoch {ep+1}/{epochs}  loss {tot/max(n,1):.3f}")
    model.eval()
    return model
 
 
def predict(model, p):
    with torch.no_grad():
        h = model.embed(p.T["feats"], p.spec.offset, p.spec.n_nodes, p.T["src"], p.T["dst"], p.T["rel"], p.T["attr"])
        return torch.sigmoid(model(h)).numpy()               # N x R object scores
 
 
def priors(P, train_idx, nR, a=2.0):
    Y = np.stack([P[i].y for i in train_idx])               # train projects x rules
    lab = Y >= 0
    g = (Y == 1).sum() / max(lab.sum(), 1)
    nok, n = (Y == 1).sum(0), lab.sum(0)
    return g, (nok + a * g) / (n + a)
 
 
def boot_ci(y, s, fn, B=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y))
        if 0 < y[ix].sum() < len(ix):
            v.append(fn(y[ix], s[ix]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (float("nan"), float("nan"))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--projects", default=None)
    ap.add_argument("--out", default="runs/real"); ap.add_argument("--exclude_rules", default="")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    log = lambda s: print(s, flush=True)
    master, P, rules = load_all(discover(a.data, a.projects),
                                exclude_rules=[r for r in a.exclude_rules.split(",") if r])
    nP, nR = len(P), len(rules)
    Y = np.stack([p.y for p in P])
    # ---------------- what the data can and cannot support (the "gate")
    lab = Y >= 0
    mixed = [r for j, r in enumerate(rules) if (Y[:, j] == 1).any() and (Y[:, j] == 0).any()]
    ext = {r for p in P for r, y, e in zip(p.labels["rule"], p.labels["y"], p.labels["ext_hint"]) if y == 1 and e}
    log(f"[gate] projects={nP} rules={nR} labelled pairs={int(lab.sum())} NOK pairs={int((Y==1).sum())}")
    log(f"[gate] rules with BOTH OK and NOK outcomes: {len(mixed)}  (plan asked for >= 30; NOK pairs >= 40)")
    log(f"[gate] NOK rules whose comment mentions SyCR/RTI/customer documents ({len(ext)}): {sorted(ext)}")
    log("[gate] check these: a rule that needs a customer value cannot be decided from the database alone; "
        "drop it with --exclude_rules if so")
    # ---------------- leave-one-project-out
    rows = []
    for k in range(nP):
        tr = [i for i in range(nP) if i != k]
        log(f"[fold {k+1}/{nP}] hold out {P[k].id}; train on {[P[i].id for i in tr]}")
        g, rp = priors(P, tr, nR)
        S = np.zeros((a.seeds, P[k].spec.n_nodes, nR))
        for s in range(a.seeds):
            m = fit(P, tr, rules, master, seed=s, epochs=a.epochs, d=a.d, lr=a.lr, log=log if s == 0 else None)
            S[s] = predict(m, P[k])
        pooled = S.max(1).mean(0)                              # rule score = max over objects, mean over seeds
        rng = np.random.default_rng(k)
        for j, r in enumerate(rules):
            if P[k].y[j] >= 0:
                rows.append(dict(project=P[k].id, rule=r, y=int(P[k].y[j]), random=rng.random(), global_rate=g,
                                 rule_rate=rp[j], gnn=float(pooled[j])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    # ---------------- metrics
    res, lines = {}, []
    y = df["y"].to_numpy()
    lines += [f"REAL-LABEL RESULT: leave-one-project-out over {nP} projects, {len(df)} labelled (project, rule) pairs, "
              f"{int(y.sum())} NOK.", "Rule score = max over objects of the object score, mean of %d seeds." % a.seeds, ""]
    lines.append(f"{'method':14s} {'AUROC':>7s} {'95% CI':>15s} {'AUPRC':>7s} {'95% CI':>15s} {'Brier':>7s}")
    for m in ("random", "global_rate", "rule_rate", "gnn"):
        s = df[m].to_numpy()
        au, ap_ = roc_auc_score(y, s), average_precision_score(y, s)
        ci1, ci2 = boot_ci(y, s, roc_auc_score), boot_ci(y, s, average_precision_score)
        br = brier_score_loss(y, np.clip(s, 0, 1))
        res[m] = dict(auroc=au, auprc=ap_, brier=br)
        lines.append(f"{m:14s} {au:7.3f} {ci1[0]:6.2f}-{ci1[1]:5.2f} {ap_:7.3f} {ci2[0]:6.2f}-{ci2[1]:5.2f} {br:7.3f}")
    lines.append(f"(NOK base rate = {y.mean():.3f}; AUPRC of a useless method is about this value)")
    lines += ["", "Per held-out project (AUROC, only where both outcomes exist):"]
    for pid, d in df.groupby("project"):
        if 0 < d["y"].sum() < len(d):
            lines.append(f"  {pid:10s} n={len(d):2d} NOK={int(d['y'].sum())}  rule_rate {roc_auc_score(d['y'], d['rule_rate']):.2f}"
                         f"  gnn {roc_auc_score(d['y'], d['gnn']):.2f}")
        else:
            lines.append(f"  {pid:10s} n={len(d):2d} NOK={int(d['y'].sum())}  (single outcome, AUROC undefined)")
    lines += ["", "Held-out NOK pairs and how each method scored them (higher = more suspicious):"]
    for _, r in df[df.y == 1].sort_values("gnn", ascending=False).iterrows():
        lines.append(f"  {r.project:9s} {r.rule:28s} gnn {r.gnn:.2f}  rule_rate {r.rule_rate:.2f}")
    open(os.path.join(a.out, "results.txt"), "w").write("\n".join(lines))
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    log("\n".join(lines))
    # ---------------- final model on ALL projects, for score.py
    log("[final] training on all projects ...")
    m = fit(P, list(range(nP)), rules, master, seed=0, epochs=a.epochs, d=a.d, lr=a.lr, log=log)
    torch.save(dict(state=m.state_dict(), d=a.d, rules=rules), os.path.join(a.out, "model.pt"))
    pickle.dump(master, open(os.path.join(a.out, "spec.pkl"), "wb"))
    log(f"done in {time.time()-t0:.0f}s; saved model.pt, spec.pkl, predictions.csv, results.txt in {a.out}")
 
 
if __name__ == "__main__":
    main()
