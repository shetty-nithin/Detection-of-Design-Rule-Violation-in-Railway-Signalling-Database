import torch, torch.nn as nn, torch.nn.functional as F
 
 
class RGCNLayer(nn.Module):
    """R-GCN with basis decomposition and a scalar edge attribute (signed ln-distance)."""
 
    def __init__(self, d, n_rel, n_bases=8):
        super().__init__()
        self.B, self.d = n_bases, d
        self.V = nn.Parameter(torch.randn(n_bases, d, d) * (d ** -0.5))
        self.coef = nn.Parameter(torch.randn(n_rel, n_bases) * 0.3)
        self.ea = nn.Parameter(torch.randn(n_rel, d) * 0.1)          # edge-attr weight per relation
        self.self_lin = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
 
    def forward(self, h, src, dst, rel, attr, deg_inv):
        N = h.size(0)
        hb = torch.einsum("nd,bde->nbe", h, self.V)                   # N x B x d
        w = self.coef[rel]                                           # E x B
        msg = (hb[src] * w.unsqueeze(-1)).sum(1) + attr.unsqueeze(1) * self.ea[rel]
        agg = torch.zeros(N, self.d, device=h.device).index_add_(0, dst, msg) * deg_inv.unsqueeze(1)
        return self.norm(h + F.relu(agg + self.self_lin(h)))
 
 
class RuleGNN(nn.Module):
    def __init__(self, dims, n_rel, n_rules, d=64, layers=2, zr=16, dropout=0.1):
        super().__init__()
        self.tables = list(dims)
        self.enc = nn.ModuleDict({t: nn.Linear(k, d) for t, k in dims.items()})
        self.layers = nn.ModuleList([RGCNLayer(d, n_rel) for _ in range(layers)])
        self.rule_emb = nn.Embedding(n_rules, zr)                       # learned rule vector z_r
        self.film = nn.Linear(zr, 2 * d)                                # FiLM(z_r) -> gamma, beta
        self.head = nn.Sequential(nn.Linear(d + zr, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.n_rules = n_rules
        self.d = d
 
    def embed(self, feats, offsets, N, src, dst, rel, attr):
        h = torch.zeros(N, self.d)
        for t in self.tables:
            x = feats[t]
            o = offsets[t]
            h[o:o + x.size(0)] = F.relu(self.enc[t](x))
        deg = torch.zeros(N).index_add_(0, dst, torch.ones(dst.numel()))
        deg_inv = 1.0 / deg.clamp(min=1.0)
        for L in self.layers:
            h = L(h, src, dst, rel, attr, deg_inv)
        return h
 
    def forward(self, h):
        """h: N x d  ->  logits N x R (one score per object per rule)."""
        outs = []
        for r in range(self.n_rules):
            z = self.rule_emb.weight[r]
            g, b = self.film(z).chunk(2)
            hr = h * (1 + g) + b
            outs.append(self.head(torch.cat([hr, z.expand(h.size(0), -1)], 1)).squeeze(1))
        return torch.stack(outs, 1)
