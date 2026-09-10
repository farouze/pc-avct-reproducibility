# Reproducibility scope

## What was rerun

The V38 analysis rebuilt the AuroraBP dose cohorts from hashed feature banks and refitted every primary context arm under three repeats of five participant folds. It also refitted the additional one- and two-reading calibration-dose arms. Each participant's outer-fold errors were averaged across repeats before paired inference. The two primary sign-flip tests were adjusted with Holm's procedure.

The run saved row predictions, participant contrasts, selected features, and gates locally. The public repository exposes aggregate statistics and input hashes. It excludes participant-level files under the dataset access conditions.

## What the attached notebooks add

The author-supplied LEAN notebook contains the historical V26 method and stored aggregate results. Syntax-tree comparison found all 14 V26 functions extracted for the V38 runner to be identical to their notebook definitions. This supports source-code provenance.

The original LEAN outputs and the V38 reference run differ slightly. For example, the historical wearable-state SBP MAE is 7.687 mmHg and the V38 reference value is 7.697 mmHg. Function identity alone cannot attribute that difference. The notebook installs unpinned current dependencies except for pandas 2.2.2, runs in Colab, and does not preserve a complete lockfile or raw-extraction manifest. The V38 reference environment is therefore reported separately and its regenerated values are used in the revised paper.

The Monk/Fitzpatrick Sub5 notebook is useful development evidence for optical-quality and pigmentation analyses. Its under-5-mmHg candidate search is exploratory and must not replace the frozen V26 primary endpoint or be described as confirmatory validation.

## What was not rerun

Raw AuroraBP waveform extraction was not repeated in V38. The feature banks are treated as inputs, with SHA-256 verification. The 500 Hz and CSI checks use the existing timestamp/feature audit. A third party who has the raw dataset still needs to regenerate the feature banks to complete an end-to-end reproduction.

The repository's continuous-integration job runs provenance and unit checks without restricted data. It does not reproduce the clinical results on GitHub.
