# Leakage before/after

| mode | top1 | note |
|------|------|------|
| no_context | 0.29770992366412213 |  |
| declared | 0.0916030534351145 |  |
| oracle_label | 0.4351145038167939 | 누설 포함 상한 |
| legacy_full_leak | 0.4351145038167939 | 누설 포함 상한 (oracle_label + test-fit hard range) |
