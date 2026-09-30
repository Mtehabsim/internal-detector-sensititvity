# Analysis designs written before the runs

Three analyses added on 2026-09-26 were specified in writing before they were run:
- the WildChat benign set (section C);
- partial AUROC (section B);
- the 30 GradSafe reference sets (section A);
- and, later that day, the same noise criterion applied to MTK's bank draws (section D).

The document lives in the authors' research record, not in a public registry. When each version was written, its
SHA-256 was recorded with a timestamp:

| File | Recorded at | Contents |
|---|---|---|
| `design_2026-09-26T17-03.md` | 2026-09-26T17:03:40-04:00 | sections A–C, before any WildChat prompt was selected, before the 24 new GradSafe draws (refset5–refset28) were scored, and before the partial-AUROC analysis was run; five earlier redraws (refset0–refset4) already existed |
| `design_2026-09-26T20-34.md` | 2026-09-26T20:34:50-04:00 | the same document with the outcomes of A–C appended, and section D before it was computed |
| `design_with_outcomes.md` | — | the current document, with every outcome appended; the designs above are unchanged prefixes of it |

Check the hashes with `sha256sum -c SHA256SUMS`. The factor-of-two rule in sections A and D is a criterion we chose
in advance (between-set spread against prompt-sampling spread), not a conventional significance test.
