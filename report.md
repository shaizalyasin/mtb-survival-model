# Overall survival prediction from cBioPortal data: results

Trained on MSK-MET (`msk_met_2021`), restricted to features that can be
retrieved for a single patient at the moment of sequencing. Seed 20261111,
80/20 split stratified by OncoTree code and event status, test split touched
once.

## Cohort

| | |
| --- | --- |
| Patients in the study | 25,775 |
| Dropped, `OS_MONTHS` missing | 116 |
| Dropped, `OS_MONTHS` at or below zero | 1,098 |
| Used | 24,561 |
| Train / test | 19,648 / 4,913 |
| Events (deaths) | 8,077 train, 2,018 test (41%) |
| Median follow-up | 17.4 months, maximum 80.9 |
| Features | 752 |

Every patient had exactly one sample, so the one-sample-per-patient rule
dropped nobody. The 1,098 patients with non-positive survival time were
sequenced at or after their last recorded contact; Cox cannot encode a
zero-length interval, so they go before the split rather than after.

**The survival clock starts at sequencing.** The study's data dictionary says
"Overall survival in months since date of sequencing", and reconstructing the
interval independently from `AGE_AT_SEQUENCING` against `AGE_AT_DEATH` or
`AGE_AT_LAST_CONTACT` agrees for 95.6% of patients within one month, median
difference 0.04 months. Features and time origin therefore coincide, which is
the condition under which this task is well posed at all.

## Headline result

| | |
| --- | --- |
| **Test C-index** | **0.7126** |
| Out-of-fold C-index | 0.7040 |
| Train C-index | 0.7336 |
| Time-dependent AUC, 12 months | 0.742 |
| Time-dependent AUC, 24 months | 0.752 |
| Time-dependent AUC, 60 months | 0.687 |
| Integrated Brier score | 0.1985 |

The integrated Brier score compares against 0.25, which is what always
predicting 50% survival would score. The drop in AUC at 60 months reflects
thinning follow-up rather than a change in model behaviour: few patients are
still at risk that far out.

## The ablation

This is the result worth arguing about.

| arm | features | C-index | 24-month AUC | Brier |
| --- | ---: | ---: | ---: | ---: |
| Clinical only (age, sex, cancer type, sample type) | 57 | 0.6733 | 0.714 | 0.1933 |
| Genomic only (mutations, CNA, TMB, MSI, FGA) | 695 | 0.6755 | 0.706 | 0.2184 |
| **Full model (served)** | **752** | **0.7126** | **0.752** | **0.1985** |
| Full + metastasis count (never served) | 753 | 0.7379 | 0.785 | 0.1838 |

| | |
| --- | ---: |
| Genomics on top of clinical | **+0.0393** |
| Clinical on top of genomic | +0.0371 |
| Metastasis count on top of full | +0.0253 |

Three things follow.

**Genomics is not decoration.** 695 genomic features add 0.039 C-index over
knowing age, sex, cancer type and sample type. That is a modest but real
contribution, and it is the number to quote when someone asks whether the
sequencing adds anything over the diagnosis.

**The two halves are roughly interchangeable in strength.** Genomic alone
(0.6755) performs about as well as clinical alone (0.6733), and each adds
roughly 0.037 to the other. The common objection that a pan-cancer model only
learns the diagnosis turns out to be half the story: the diagnosis is worth
about as much as the entire gene panel, not more.

**One excluded feature is worth two thirds of the whole gene panel.**
Metastasis count, a single column, adds 0.0253. The 695 genomic features add
0.0393. That single column is why the published figure is higher than ours, and
it is unavailable for a newly sequenced patient because it counts metastatic
sites accumulated across the entire disease course. A model using it would
score better and tell an oncologist less.

## Comparison with the published result

[Mestrallet, *Onco* 2025](https://doi.org/10.3390/onco5010008) reports a
C-index of 0.76 on a Random Survival Forest, and 74% accuracy for binary
overall survival status.

Our 0.7126 is lower, by construction, for reasons that are both deliberate:

1. The 0.76 was measured on `msk_met_2021` merged with `msk_ch_2020`
   (n = 16,603) **with treatment as a feature**. Treatment is not knowable when
   a new patient is sequenced.
2. Its strongest single predictor is metastasis count, which we exclude as
   future information. Our arm 4 measures what that exclusion costs: 0.0253.

The 74% accuracy figure is not comparable to a C-index at all. It is binary
classification of living against deceased, which discards censoring: a patient
alive at three months of follow-up carries the same label as one alive at eight
years.

Adding back the one feature we can measure takes us to 0.7379, which closes
roughly half the gap to 0.76. The remainder is plausibly treatment data and
model family. We have not tried to close it, because a model that needs
treatment history cannot score a patient at the point where the decision is
actually made.

## Per cancer type

Pan-cancer concordance is partly the model ranking cancer types against each
other. Within a single type that advantage disappears. Full model, test split,
types with at least 100 test patients:

| OncoTree | n | events | C-index |
| --- | ---: | ---: | ---: |
| LUAD | 789 | 339 | 0.6906 |
| COAD | 445 | 165 | 0.6375 |
| IDC | 433 | 157 | 0.7473 |
| PRAD | 409 | 114 | 0.7651 |
| PAAD | 330 | 223 | 0.6088 |
| HGSOC | 190 | 72 | 0.5692 |
| BLCA | 181 | 58 | 0.6417 |
| UEC | 162 | 25 | 0.7088 |
| READ | 151 | 49 | 0.6534 |
| SKCM | 133 | 41 | 0.6237 |
| LUSC | 104 | 57 | 0.6866 |

Within-type performance ranges from 0.57 to 0.77 and averages well below the
pan-cancer 0.7126. High-grade serous ovarian at 0.5692 is barely better than
chance, and pancreatic at 0.6088 is weak. This is the honest view of what the
model does for an individual patient, and it is why the tab reports a
percentile against the patient's own cancer type rather than a pan-cancer
score.

## Two data findings that changed the model

**Off-panel genes are untested, not wild type.** MSK-MET mixes three IMPACT
panel versions (IMPACT468 68.3%, IMPACT410 24.7%, IMPACT341 7.0%, strictly
nested). A gene not on the panel a given sample was sequenced with was never
looked at, so coding it zero asserts a negative result nobody measured. Those
cells are NaN, which is 9.35% of all gene cells. Panel definitions are not in
the study tarball and are fetched from the cBioPortal API and cached.

**MSI and TMB are complete in this cohort.** The specification assumed
meaningful missingness and provided for `_missing` indicator columns. In
MSK-MET both are 100% present, so those indicators are constant zero in
training. They are retained because the serving path needs them, but it follows
that the model has never seen a missing MSI during training and its behaviour
when a served study lacks one is determined by XGBoost's default branch rather
than by evidence. The response reports which features were absent so this is
visible rather than silent.

## Hyperparameter search, and why nothing changed

120 random configurations plus the incumbent, each scored by 5-fold
out-of-fold C-index on the training split alone. The test split was read once,
at the end, for the defaults and the winner only.

| | OOF C-index | depth | lr | subsample | colsample | min_child_weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trial 82 (best) | 0.7062 +/- 0.0078 | 8 | 0.026 | 0.62 | 0.44 | 1 |
| trial 9 | 0.7055 +/- 0.0097 | 6 | 0.027 | 0.83 | 0.68 | 5 |
| trial 47 | 0.7054 +/- 0.0085 | 5 | 0.022 | 0.88 | 0.56 | 1 |
| **defaults (shipped)** | **0.7039 +/- 0.0083** | 4 | 0.050 | 0.80 | 0.80 | 10 |

Across all 121 trials: best 0.7062, median 0.7019, worst 0.6926. Nineteen beat
the defaults.

**The defaults were kept.** The winner's margin is +0.0022, which is smaller
than the +/-0.0078 spread between its own folds, so the ranking is not stable
enough to call it a better model. The test split agrees that nothing much is
there: 0.7126 for the defaults against 0.7136 for the winner, a gain of
+0.0010 on 4,913 patients.

This is worth stating plainly because it is the useful finding. The entire
hyperparameter space is worth about a fifth of what adding genomic features to
clinical ones was worth (+0.0393), and less than a tenth of what the single
metastasis-count column was worth (+0.0253). On this cohort the model is
limited by what it is allowed to know about the patient, not by how it is
fitted. Tuning harder is not the lever.

## Limitations

**Immortal time.** Patients had to survive long enough to be sequenced and
enrolled. The time origin is sequencing, so features and clock coincide, but
entry into the cohort is still conditioned on surviving to that point. The
model therefore describes prognosis among patients who reached sequencing, not
among all patients with the diagnosis.

**Single institution, single panel family.** Trained entirely on MSK-IMPACT.
Panel gene coverage and germline filtering differ across commercial and
institutional assays. Transferring to another site's panel requires checking
the overlapping gene set and setting genes outside it to missing rather than
zero. The machinery for that exists and is exercised across the three IMPACT
versions, but not across vendors.

**No treatment data.** The model predicts prognosis under whatever care the MSK
cohort received between roughly 2014 and 2021. It is not a treatment-effect
model and must not be read as one. It cannot answer "will this therapy help".

**Cancer-type dominance, quantified above.** Clinical features alone reach
0.6733. The genomics adds 0.0393 on top. Anyone expecting the sequencing to
dominate the prediction should look at that number first.

**Cohort selection.** MSK-MET is a metastatic-tropism cohort. `SAMPLE_TYPE`
distinguishes which site was biopsied, not whether the patient has metastatic
disease, and generalisation to newly diagnosed localised disease is outside
what this cohort can support.

**Not prospectively validated.** Research use only, not for clinical
decision-making. A model whose output informs treatment decisions is likely to
fall under MDR, a question that needs settling before any use beyond research.

## Reproducing

```bash
make setup
make train        # ~3 minutes
make ablation     # ~15 minutes, all four arms
make test
```

Seed, split and hyperparameters are recorded in `artifacts/model_card.json`;
the ablation writes `artifacts/ablation.json`.
