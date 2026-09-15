# Table 7 reproduction record

## Status

`ACCEPTED_WITH_DOCUMENTED_SCOPE` under the policy in `docs/ACCEPTANCE.md`
(internally labeled `EXACT`; the legacy word remains only in paths such as
`results/main_exact_rerun/`).

All four `(m,n)=(10,10)` OOD experiments were regenerated with the recovered
final data and all ten base checkpoints. Each variant produced 100 sample-level
seed-averaged rows and 1,000 seed-by-sample rows.

## Non-runtime comparison

| Variant | Regenerated mean/std | Paper mean/std | Result |
|---|---:|---:|---|
| `log(1+x)` | `0.980 / 0.010` | `0.980 / 0.010` | match |
| `x^(1/3)` | `0.950 / 0.019` | `0.950 / 0.019` | match |
| `Beta(5,5)` | `0.987 / 0.008` | `0.987 / 0.008` | match |
| `Beta(.5,.5)` | `0.991 / 0.008` | `0.991 / 0.008` | match |

For `Beta(.5,.5)`, the raw population standard deviation is
`0.007491434479918724` and the raw sample standard deviation is
`0.007529174942855045`. The paper's three-decimal `0.008` agrees with the sample
standard deviation; this statistical convention difference was explicitly
accepted as negligible. Both raw values remain recorded.

Runtime columns are kept in the output CSV files but excluded from acceptance.

## Commands

```bash
uv run python scripts/reproduce.py replay table7
uv run python scripts/reproduce.py verify table7
# Equivalent lower-level verifier:
uv run python scripts/verify_main_results.py --experiment table7
```
