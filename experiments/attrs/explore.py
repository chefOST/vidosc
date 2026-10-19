"""Scratch: evidence/decoder variants on dynamic tracks (bacc) for a schema/encoder."""
import sys, json, numpy as np
import evaluate as Ev, method as M
sch = sys.argv[1] if len(sys.argv) > 1 else 's0'
schema = json.load(open(f'schemas/schema_{sch}.json')); emap = json.load(open(f'schemas/evalmap_schema_{sch}_claude-opus-5-5.json'))
for enc in M.ENCODERS:
    tr = [t for t in Ev.load_tracks(schema, emap, enc) if Ev.dynamic(t)]
    def ev_nogate(t): S = t['S']; return M.zc(M.smooth(S - S.mean(1, keepdims=True)))
    def ev_raw(t): S = t['S']; return M.smooth(S - S.mean(1, keepdims=True)) * 100
    V = {'prior': lambda t: M.prior_split(t['T'], t['script']),
         'zs argmax': lambda t: M.frame_argmax(t['S']),
         'soft gated-z': lambda t: M.script_soft(t['E'], t['script']),
         'soft nogate-z': lambda t: M.script_soft(ev_nogate(t), t['script']),
         'soft raw(x100)': lambda t: M.script_soft(ev_raw(t), t['script']),
         'hard nogate-z': lambda t: M.script_dp(ev_nogate(t), t['script']),
         'hard raw': lambda t: M.script_dp(ev_raw(t), t['script']),
         'viterbi raw': lambda t: M.monotone_free(ev_raw(t), t['one_way']),
         }
    print(enc, len(tr), '  '.join(f'{k}={np.mean([Ev.track_scores(t, f(t))[1] for t in tr]):.3f}' for k, f in V.items()))
