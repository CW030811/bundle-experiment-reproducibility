# Portability patches to the final sources

`src/deterministic/` is exported from the final internal `Bundle_code/` sources.
The following compatibility changes let those sources run in the pinned
PyTorch 2.8 / PyG 2.6 environment. None of them changes an experiment algorithm.

1. Legacy checkpoints pickle the whole `__main__.EdgeScoringGCN` object.
   PyTorch 2.6+ defaults to `weights_only=True`, so the original loader fails.
   The legacy class name is registered before loading, and the trusted
   checkpoints shipped in this package are loaded with an explicit
   `weights_only=False`.
2. The PCP cover DP can pass a `numpy.int64` mask to `.bit_count()`, which does
   not exist on that type or on Python 3.9 integers. It is replaced by
   `bin(int(mask)).count("1")`, which yields the same popcount and sort key.
3. The Table 5 scripts only changed in-package default paths. The Z fix,
   instance generation, model, seeds and solver logic are unchanged.
4. `src/appendix/rerun_seed1.py` generates the paper's Figure 9 and Figure 10.
   It was rebuilt from the seed-10 generic runner and set to the final
   provenance: seed 1, 30 FCP samples, 10 separate PCP samples, 30 K samples
   and a 600-second limit per solve. Its output is the reference data.
5. The Appendix E runner, cached-LP source and results were recovered from the
   project history. The public version changes the data-directory default to the
   current directory and the plotting cache to an in-package path; solver logic is
   unchanged. Original and public source hashes are recorded in
   `SOURCE_EXPORT.json`, and the guard uses the public hash
   (`FIGURE11_SOURCE_GUARD.md`). The replot script only switches its JSON and
   output directories to in-package relative paths.
6. The unified public entry point adds experiment selection, separate output
   directories, fixed sample-list checks, complete seed/sample checks and
   data-driven plotting. Legacy evaluator defaults point to the final public
   data and models, and model paths in training summaries are normalized
   accordingly.
7. `data_pipeline.py` unifies generation, labeling and training paths, links the
   random-data train/eval/test manifests, and sets a headless plotting backend
   for tmux and CI.
8. Chinese comments, docstrings, argparse help, input prompts and console
   messages were translated to English. A token-level comparison confirmed that
   only comment and string text changed; identifiers, operators, numbers and
   f-string expressions are identical. The two SHA-256 guarded cached-LP files in
   `src/test/` are intentionally left byte-identical.
9. The Figure 11 archive extracts to `data/appendix_e/` and
   `models/appendix_e_seed1/` instead of the legacy `Dataset/` and
   `models_multi_layer_edge_update/` directories. The runner, verifier and
   archived Appendix E manifest were updated; input and checkpoint bytes are
   unchanged. Standalone defaults in `src/appendix/legacy_runtime/` point to the
   packaged Figure 9/10 inputs and checkpoint.

Key SHA-256 prefixes of the original `Bundle_code` files:

| File | SHA-256 prefix |
|---|---|
| `Training_multi_layer.py` | `503731a08a1904c1` |
| `generate_data_bundle.py` | `3a1fbee1b272bbb3` |
| `generate_data_PCP_cp.py` | `0297b6405910de28` |
| `test_FCP.py` | `2ac79f45badf2bcf` |
| `test_PCP_cp.py` | `be1a6e84b79ad678` |
| `test_FCPLS_score_cached_lp.py` | `f84359f2e1fb7276` |
| `test_BSP.py` | `410388014ffa88d8` |

For run verification, see `docs/ACCEPTANCE.md` and
`scripts/verify_reference_results.py`.
