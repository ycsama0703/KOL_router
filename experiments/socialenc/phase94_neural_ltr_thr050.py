"""Phase94: neural listwise rankers vs GBDT on {context+O_k}, single window 25.6-26.6.
- MLP listwise (per-candidate MLP, ListNet loss) : isolates neural without cross-candidate attention
- Context-aware (self-attention across <=10 candidates per event, ListNet loss) : the SOTA-frontier ranker
Compare to LambdaMART 0.811 / ridge 0.745 references.
thr=0.50 first10 reach.
"""
from __future__ import annotations
import collections, json, math, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import phase5_sentiment_reconstruction as p5
import phase7_origin_alert as p7
import phase33_origin_alert_ablation as p33
import phase65_pit_lightweight_2025_2026 as p65

THR = 0.50; MEK = 8; TARGET = "log_future_reach"
TR_S, TE_S, TE_E = "2024-06-01", "2025-06-01", "2026-06-01"; INNER = "2025-01-01"
MAXC = 10; DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
def log(m): print(m, flush=True)
p65.split_for_block = lambda b: b; p65.THR = THR
def block_for_day(day):
    if TR_S <= day < TE_S: return "train"
    if TE_S <= day < TE_E: return "test"
    return None
p65.block_for_day = block_for_day
def states_for(rows_by, emb_by, all_rows):
    states = {}
    for block, cutoff in [("train", TR_S), ("test", TE_S)]:
        hist_rows = p65.rows_before(all_rows, cutoff); meta = p5.compute_metadata(hist_rows)
        ol = p65.with_train_end(cutoff, p5.compute_oltrait, hist_rows, meta)
        hist = p65.with_train_end(cutoff, p7.compute_origin_history, rows_by, emb_by, THR)
        raw_ol = p65.with_train_end(cutoff, p33.compute_raw_oltrait, hist_rows, meta)
        states[block] = {"cutoff": cutoff, "meta": meta, "ol": ol, "hist": hist, "raw_ol": raw_ol, "shuffled_ol": p65.make_shuffled_ol(ol, block)}
    return states

class MLPRanker(nn.Module):
    def __init__(self, f, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, 1))
    def forward(self, x, mask):
        return self.net(x).squeeze(-1)

class CtxRanker(nn.Module):
    def __init__(self, f, d=64, heads=4):
        super().__init__()
        self.inp = nn.Linear(f, d)
        self.attn = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d, dropout=0.2, batch_first=True)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))
    def forward(self, x, mask):
        h = self.inp(x)
        h = self.attn(h, src_key_padding_mask=~mask)
        return self.out(h).squeeze(-1)

def listnet_loss(scores, y, mask):
    neg = torch.full_like(scores, -1e9)
    s = torch.where(mask, scores, neg)
    ylog = torch.where(mask, y, neg)
    p = torch.softmax(ylog, dim=1)
    logq = torch.log_softmax(s, dim=1)
    return -(p * torch.where(mask, logq, torch.zeros_like(logq))).sum(dim=1).mean()

def build_events(panel, X, split_sel):
    by = collections.OrderedDict()
    for i, r in enumerate(panel):
        if r["split"] in split_sel: by.setdefault(r["event_id"], []).append(i)
    E = len(by); F = X.shape[1]
    feat = np.zeros((E, MAXC, F), np.float32); yv = np.full((E, MAXC), -1e9, np.float32)
    msk = np.zeros((E, MAXC), bool); idxmap = [[-1] * MAXC for _ in range(E)]
    for e, (ev, idxs) in enumerate(by.items()):
        for j, i in enumerate(idxs[:MAXC]):
            feat[e, j] = X[i]; yv[e, j] = float(panel[i][TARGET]); msk[e, j] = True; idxmap[e][j] = i
    return feat, yv, msk, idxmap

def symbal_ndcg_from_pred(panel, pred, split):
    evs = p65.event_rows_for_split(panel, pred, TARGET, split)
    return p7.symbal_mean(evs, "ndcg3")

def train_eval(panel, X, ModelCls):
    finite = np.array([r[TARGET] for r in panel], dtype=float); finite = np.isfinite(finite)
    f_tr, y_tr, m_tr, _ = build_events(panel, X, {"train"})
    # inner split by day on train events
    day = {r["event_id"]: r["day"] for r in panel}
    fe, ye, me, imap = build_events(panel, X, {"train"})
    evids = list(collections.OrderedDict((r["event_id"], 1) for r in panel if r["split"] == "train").keys())
    is_dev = np.array([day[ev] >= INNER for ev in evids])
    Ft = torch.tensor(fe, device=DEV); Yt = torch.tensor(ye, device=DEV); Mt = torch.tensor(me, device=DEV)
    fit = ~is_dev
    Ffit = Ft[fit]; Yfit = Yt[fit]; Mfit = Mt[fit]
    model = ModelCls(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # test events for monitoring
    fte, yte, mte, imte = build_events(panel, X, {"test"})
    Fte = torch.tensor(fte, device=DEV); Mte = torch.tensor(mte, device=DEV)
    best = (-1.0, None); patience = 0
    for epoch in range(200):
        model.train()
        perm = torch.randperm(Ffit.shape[0], device=DEV)
        for s in range(0, len(perm), 256):
            b = perm[s:s + 256]
            opt.zero_grad()
            sc = model(Ffit[b], Mfit[b])
            loss = listnet_loss(sc, Yfit[b], Mfit[b])
            loss.backward(); opt.step()
        # eval indev NDCG
        model.eval()
        with torch.no_grad():
            sc_all = model(Ft, Mt).cpu().numpy()
        pred = np.full(len(panel), np.nan)
        for e in range(len(imap)):
            if not is_dev[e]: continue
            for j in range(MAXC):
                if imap[e][j] >= 0: pred[imap[e][j]] = sc_all[e, j]
        nd = symbal_ndcg_from_pred(panel, pred, "train")
        if nd > best[0]:
            best = (nd, {k: v.detach().clone() for k, v in model.state_dict().items()}); patience = 0
        else:
            patience += 1
            if patience >= 25: break
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        sc_te = model(Fte, Mte).cpu().numpy()
    pred = np.full(len(panel), np.nan)
    for e in range(len(imte)):
        for j in range(MAXC):
            if imte[e][j] >= 0: pred[imte[e][j]] = sc_te[e, j]
    evs = p65.event_rows_for_split(panel, pred, TARGET, "test")
    return {k: p7.symbal_mean(evs, k) for k in p7.METRICS}, best[0]

def main():
    t0 = time.time()
    rows_by, emb_by, all_rows = {}, {}, []
    for sym in p5.SYMS:
        r, e = p5.load_symbol(sym); rows_by[sym] = r; emb_by[sym] = e; all_rows.extend(r)
    p7.MIN_EVENT_KOLS = MEK; p65.TRAIN_START = TR_S; p65.TEST_END = TE_E
    states = states_for(rows_by, emb_by, all_rows)
    p65.ORIGIN_WINDOW = {"name": "first10", "max_rank": 10}
    panel, _ = p65.build_pit_origin_panel(rows_by, emb_by, states)
    tr = np.array([r["split"] == "train" for r in panel])
    feats = p7.FEATURE_SETS["no_ol_strong"] + ["origin_ol"]
    Xraw = p65.matrix(panel, feats)
    med, mu, sd = p65.train_standardizer(Xraw, tr)
    X = p65.apply_standardizer(Xraw, med, mu, sd).astype(np.float32)
    log("panel=%d test=%d feats=%d dev=%s" % (len(panel), (~tr).sum(), X.shape[1], DEV))
    for name, Cls in [("Neural MLP listwise", MLPRanker), ("Context-aware (self-attn) listwise", CtxRanker)]:
        m, devnd = train_eval(panel, X, Cls)
        log("  %-34s NDCG=%.4f Hit=%.4f Mass=%.4f JS=%.4f (inner-dev NDCG=%.4f)" % (name, m["ndcg3"], m["hit1"], m["mass3"], m["js"], devnd))
    log("  [ref] LambdaMART=0.811  ridge=0.745  Qwen3-4B=0.729")
    log("done %.1fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
