"""Rate variant of phase100: overload g_net slot with net-lead RATE (out-in)/(out+in).
Recovers raw o,i from log1p via expm1. Run writes phase100's result json; caller
copies it to *_rate then git-restores the original."""
import math
import phase100_graph_consolidate as p100
_orig = p100.graph_feats
def gf_rate(all_rows, cutoff):
    gf = _orig(all_rows, cutoff)
    for v in gf.values():
        o = math.expm1(v["g_out"]); i = math.expm1(v["g_in"]); t = o + i
        v["g_net"] = (o - i) / t if t > 0 else 0.0
    return gf
p100.graph_feats = gf_rate
if __name__ == "__main__":
    p100.main()
