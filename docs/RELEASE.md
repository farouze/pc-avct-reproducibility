# Release and DOI checklist

The repository is ready to publish as a code-and-provenance record. Complete these actions before citing a software DOI in the paper:

1. Both authors approve the exact public contents and choose a software license.
2. Push the prepared repository without anything under `data/`, `runs/`, or `private/`.
3. Confirm that the GitHub validation workflow passes.
4. Connect the GitHub repository to an archival service such as Zenodo.
5. Create a GitHub release, preferably tagged `v38.0.0`, and archive that release.
6. Copy the DOI issued for the archived release into `CITATION.cff`, the README, and the paper's Code Availability section.
7. Recompile the paper and verify that the software DOI is distinct from the AuroraBP dataset DOI.

Do not reserve space with a fabricated DOI or use the GitHub URL in a DOI field. Do not archive restricted feature banks, participant predictions, or participant contrasts.

The suggested repository description is:

> Reproducibility code and provenance for the V38 PC-AVCT model-analysis rerun on AuroraBP, including paired participant inference and Holm correction.

The suggested release title is `PC-AVCT reproducibility release v38.0.0`.
