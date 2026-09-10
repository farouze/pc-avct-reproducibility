# Local inputs — excluded from Git

Obtain AuroraBP through the dataset provider's access process. Do not upload the restricted data, participant features, or per-participant predictions to GitHub.

The reference model-analysis run requires two pre-extracted local feature banks:

- aurora_raw_avct_features.csv
- uci_pc_features.csv

`docs/reference_run_manifest.json` records their SHA-256 digests. The runner refuses a mismatch so that a different input bank cannot silently masquerade as the V38 reproduction.

This repository does not supply the banks. The supplied historical notebooks contain waveform extraction procedures, but these procedures were not freshly rerun from raw data in V38. Generating an equivalent input bank from newly acquired data remains an upstream reproduction step. Do not describe the feature-bank workflow as end-to-end raw-data replication.

The optional waveform-quality audit additionally requires measurement_optical_metrics.csv. Its hash and coverage are recorded in results/aggregate/signal_audit.json.
