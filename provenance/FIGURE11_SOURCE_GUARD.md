# Figure 11 source guard

Figure 11 (Appendix E) must be replayed with the cached-LP Local Search
implementation that produced the accepted 489/522 LP-to-MILP translations.
To keep a replay from silently using another implementation,
`src/appendix_rerun/run_guarded_sensitivity.py` checks the SHA-256 of both
files below before solving and stops on any mismatch.

| File | Role |
|---|---|
| `src/test/test_FCPLS_score_cached_lp.py` | Module imported by the runner |
| `src/test/test_FCPLS_score_cached_lp_1.py` | Byte-identical compatibility copy |

Expected SHA-256 for both files:

`da4e9c4942eb2b2d993d90a521a2be77952f26ede6a072272260416c577839dd`

The same value is registered as
`appendix_e_final_provenance.cached_lp_source_sha256` in
`provenance/PUBLISHED_VALUES.json`. `scripts/verify_reference_results.py`
checks it together with the SHA-256 of the seed-1 checkpoint
`models/appendix_e_seed1/best_model_edge_4layer_seed1.pt`.

Because of this guard, both files stay byte-identical to the accepted run.
The runner imports this module directly; it does not use older Local Search
implementations.

Manual check from the repository root:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
expected = 'da4e9c4942eb2b2d993d90a521a2be77952f26ede6a072272260416c577839dd'
for path in [Path('src/test/test_FCPLS_score_cached_lp.py'), Path('src/test/test_FCPLS_score_cached_lp_1.py')]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path, digest)
    assert digest == expected, (path, digest)
print('FIGURE11_SOURCE_GUARD_VERIFIED')
PY
```
