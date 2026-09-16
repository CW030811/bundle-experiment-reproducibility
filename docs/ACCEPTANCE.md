# Reproduction scope and acceptance policy

The local project calls accepted entries `EXACT`. This public
release uses `ACCEPTED_WITH_DOCUMENTED_SCOPE` to avoid implying bitwise,
end-to-end retraining equivalence where that was not established.

1. Runtime, runtime ratios, and machine-dependent timing columns are not acceptance targets.
2. Table 2 checks each final ten-seed/sample aggregation (BSP is seed-independent).
   Two means cross the nominal three-decimal rounding boundary by only about
   `1.41e-5` and `4.07e-6`. The documented tolerance is a half display unit plus
   at most `0.00005`; raw values are retained, not replaced by paper values.
3. Table 6 checks ten archived training-loss records and a fresh ten-seed FCP replay
   (40 reported non-runtime values). It is **not** evidence of newly retraining all ten models.
4. Table 7's Beta(.5,.5) population std is `0.007491434479918724`, while sample std
   is `0.007529174942855045`. The paper's `0.008` is accepted using the documented
   sample-std convention. Other mean/std values match their displayed precision.
5. Figure 9 supplies a seed-1 cutoff-sensitivity reference/replay package with
   1,080 experiment keys, nine cutoffs and a 600-second per-solve limit.
   All non-runtime fields are bit-identical between reference and replay.
6. Figure 10 has 630 identical non-runtime replay rows/fields with a
   600-second per-solve limit. All non-runtime fields are bit-identical.
7. Figure 11's 60 per-instance translation vectors and paths reproduce `489/522`.
   Its exact solver sources and checkpoint are SHA-256 guarded.
8. Table 5 contains 120 FCP/BSP sweep rows and 30 CPBSD-A cells: two scales,
   three cost regimes, five seeds. FCP/BSP use fixed/buggy Z variants; CPBSD-A was
   rerun under the protocol the paper states (MIPGap 1e-3, 300 s, 8 threads). The
   fixed-method InS/OOS averages reproduce 36 paper statistics. The CPBSD-A values
   this rerun replaced are kept in `provenance/PUBLISHED_VALUES.json` under
   `table5_cpbsd_a_rerun`. N=30 CPBSD-A cells reach the 300-second limit, and the
   in-sample optimum is well determined while the optimal price vector is not, so
   the out-of-sample profit is reproducible only under the same solver configuration.

`scripts/verify_reference_results.py` audits the released files. Passing this
command does not mean it just retrained a model or solved a MILP. Actual replay
commands are listed in each experiment guide. Training can vary across hardware
and library builds; archived checkpoints are the reference for numerical replay.

9. Tables 3–4 check all 17/15 reported method-size cells after complete ten-seed
   evaluation. Unreported PCP cells are not fabricated.
10. Figure 7 checks all 12 curve points. Figure 8 checks all 20 base/BSP and
    self-improved points with the appropriate model family for each curve.
11. Figure 6 checks all 8 method/dataset cells: FCP and BSP at m=20, 40, 60 and
    80, with BSP normalized by its archived label.

All twelve numerical experiments have source/data/result bundles.
`scripts/reproduce.py verify all --reference` applies these registered scopes.
Plotting hard-coded paper arrays alone is not an independent replay;
the unified plot command reads the selected result CSVs/JSONs.
