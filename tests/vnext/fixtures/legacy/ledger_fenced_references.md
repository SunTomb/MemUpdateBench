# Fenced ledger references

```text
tests/vnext/fixtures/legacy/p83_conflict_rows.csv
results/**/*.json
```

```bash
python scripts/missing_audit.py \
  --input tests/vnext/fixtures/legacy/p83_conflict_rows.csv \
  --output results/missing_fenced_summary.json
```

Prose such as results/mentioned-but-not-code.json must not be audited.
