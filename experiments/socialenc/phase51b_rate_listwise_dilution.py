"""Phase51b: rate-normalized variant of phase51.

Identical to phase51 (same window/panel/LLM/capture metric, same DeepSeek cache)
EXCEPT the graph feature fed to the "ol" router is the volume-normalized
net-lead RATE  g_net_rate = (out-in)/(out+in)  instead of the raw net-degree
g_net = out-in. We do this by overloading the "g_net" column with the rate
value, so the existing feat_ol = base+["origin_ol","g_net"] transparently uses
the rate and NOTHING else changes. Separate OUT; reuses phase51's LLM cache
(cache keys include candidate_indices, so only changed ol shortlists add keys).
"""
from __future__ import annotations
import pathlib
import phase98_graph_struct as g
import phase106_gnet_activity_confound as p106
import phase51_graph_listwise_dilution as p51


def graph_feats_rate(all_rows, cutoff):
    gf = p106.graph_feats2(all_rows, cutoff)
    for k in gf:
        gf[k]["g_net"] = gf[k]["g_net_rate"]  # overload: ol router picks up the rate
    return gf


# patch the graph machinery used by phase51.build_graph_panel / routing_scores
g.graph_feats = graph_feats_rate
g.attach_graph = p106.attach

# separate outputs (do not overwrite the raw-g_net results); reuse same LLM cache
p51.OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase51b_rate_listwise_dilution_result.json"
p51.TABLE_OUT = pathlib.Path.home() / "workspace/projects/alphagap/experiments/socialenc/phase51b_rate_listwise_dilution_table.md"

if __name__ == "__main__":
    p51.main()
