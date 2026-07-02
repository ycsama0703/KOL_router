"""Rate variant of phase103 shuffle control: overload g_net slot with rate before shuffling."""
import math
import phase103_graph_shuffle_control as p103
_orig = p103.graph_feats
def gf_rate(all_rows, cutoff):
    gf = _orig(all_rows, cutoff)
    for v in gf.values():
        o = math.expm1(v["g_out"]); i = math.expm1(v["g_in"]); t = o + i
        v["g_net"] = (o - i) / t if t > 0 else 0.0
    return gf
p103.graph_feats = gf_rate
if __name__ == "__main__":
    p103.main()
