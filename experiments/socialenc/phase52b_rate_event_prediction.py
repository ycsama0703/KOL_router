"""Phase52b: rate-normalized variant of phase52 (event prediction).

Identical to phase52 EXCEPT the "ol" graph router uses g_net_rate (overloaded
into the g_net column, exactly as phase51b). All LLM competitors, encoders,
prior-art, and the label/bootstrap 口径 are untouched (no new LLM calls: the
phase85/86 caches are per-item reach scores independent of our shortlister).
Separate OUT.
"""
from __future__ import annotations
import pathlib
import phase98_graph_struct as g
import phase106_gnet_activity_confound as p106
import phase51_graph_listwise_dilution as p51
import phase52_event_prediction as p52


def graph_feats_rate(all_rows, cutoff):
    gf = p106.graph_feats2(all_rows, cutoff)
    for k in gf:
        gf[k]["g_net"] = gf[k]["g_net_rate"]
    return gf


g.graph_feats = graph_feats_rate
g.attach_graph = p106.attach

p52.OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase52b_rate_event_prediction_result.json"
p52.TABLE_OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase52b_rate_event_prediction_table.md"

if __name__ == "__main__":
    p52.main()
