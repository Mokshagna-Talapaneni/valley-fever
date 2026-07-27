# Valley Fever Patient Grouping

A pipeline that clusters synthetic Valley Fever (coccidioidomycosis)
patients by symptom presentation and social determinants of health
(SDOH), evaluates whether those groups are statistically trustworthy and
clinically meaningful, and demonstrates how the grouping would actually
be used — scoring one new patient at a time as they come in.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic patient dataset (no real patient data used) |
| `clustering.py` | Preprocessing, clustering algorithms, evaluation metrics, cluster profiling |
| `stability.py` | Bootstrap stability analysis — are the groups reproducible? |
| `feature_importance.py` | What actually separates the clusters? (Random Forest on cluster labels) |
| `equity_audit.py` | Are clusters just re-encoding income/insurance? Quantifies it |
| `assign_new_patient.py` | Scores a single new patient against the fitted model — the real deployment use case |
| `main.py` | Runs the full batch pipeline end-to-end, saves plots/tables to `outputs/` |
| `test_pipeline.py` | Sanity tests for each pipeline stage |
| `outputs/` | Generated CSVs and PNG screenshots from the run described below |

## How to run

```bash
pip install -r requirements.txt
python main.py                  # full batch pipeline -> outputs/
python assign_new_patient.py    # score one new patient against the fitted model
```

Synthetic data uses a fixed random seed, so results are reproducible.

## 1. Design choices and assumptions

**No real data was available**, so I generated 300 synthetic patients
(`generate_data.py`) drawn from three overlapping latent archetypes —
"mild/well-resourced," "moderate/under-resourced," "severe/high-risk" —
reflecting a documented pattern in Valley Fever epidemiology where
unstable housing, weak social support, and poor access to care associate
with later, more severe presentation. Substantial noise is added so
clusters aren't trivially separable, and the clustering code never sees
the "true" archetype label — it's used only afterward, for internal
validation.

**Features:** clinical (fever, fatigue, cough, joint/chest pain, rash,
night sweats, % weight loss, symptom duration, disseminated disease) and
SDOH (housing stability, employment, education, social support,
access-to-care, insurance, distance to clinic, income).

**Preprocessing:** numeric/ordinal features are standardized (z-scored)
so no feature dominates purely by scale; nominal categoricals
(employment, education, insurance, income) are one-hot encoded rather
than label-encoded, since there's no natural numeric ordering between,
say, "employed_ft" and "unemployed."

**Algorithm:** K-Means is primary (fast, deterministic, and its
centroids are easy to explain as "the typical patient in this group"),
cross-checked against Agglomerative clustering and a Gaussian Mixture
Model — two structurally different algorithms with no spherical-cluster
assumption. Agreement between all three (Adjusted Rand Index, ARI) is
itself evidence the structure is real rather than an artifact of one
algorithm's assumptions.

**Choosing k:** rather than hard-coding k=3 to match the data-generation
process, k is selected automatically by scanning k=2–8 and maximizing
silhouette score.

**A calibration bug I caught and fixed:** my first version of the GMM
used `covariance_type='full'` (a separate covariance matrix per
cluster). I only found the problem because I built a "boundary case"
patient-scoring tool (`assign_new_patient.py`) and tested it on a point
constructed to sit *exactly* halfway between the two cluster centroids —
it should score close to 50/50, but the GMM reported 100% confidence in
one cluster. With ~30 dimensions after one-hot encoding and only ~150
patients per cluster, per-component covariance estimation was
underdetermined and produced runaway, badly-calibrated probabilities —
even ~99.7% of the *training* points were scored above 99% confidence,
which is not plausible for real overlapping clinical data. Switching to
`covariance_type='tied'` (one covariance matrix shared across
components — effectively a linear-discriminant-style boundary) fixed
it: the same midpoint test point now scores 55.8%/44.2%, and the
"boundary case" flag correctly fires. I kept the bug and the fix
documented here (also in `clustering.py`'s docstring) rather than
quietly correcting it, since catching failures like this — not just
getting code to run without erroring — is the actual work.

## 2. Results from the example run (`outputs/`)

| Metric | Value |
|---|---|
| Selected k | 2 (highest silhouette across k=2–8) |
| KMeans silhouette / Davies-Bouldin | 0.359 / 1.132 |
| ARI, KMeans vs. Agglomerative | 0.921 |
| ARI, KMeans vs. GMM | 0.921 |
| ARI, KMeans vs. hidden synthetic archetype | 0.728 |
| **Bootstrap stability** (50 resamples) | **mean ARI = 0.970 (± 0.040) → stable** |
| Random-Forest CV accuracy predicting cluster from features | 0.980 (± 0.007) |

**Silhouette peaked at k=2, not k=3** — the algorithm merged "moderate"
and "severe" into one broad "elevated risk" group rather than keeping
them separate (silhouette 0.359 at k=2 vs. ~0.31 at k=3–4). I left this
as-is rather than forcing k=3: it's a genuine, defensible finding that
should be reported to a domain expert rather than overridden to match
what I expected going in.

**Stability:** bootstrap resampling gives mean ARI = 0.970 across 50
resamples — the grouping is highly reproducible, not an artifact of one
particular sample of 300 patients (`outputs/05_bootstrap_stability.png`).

**What actually separates the clusters** (`outputs/06_feature_importance.png`,
Random Forest feature importances): `weight_loss_pct` (0.141),
`night_sweats_severity` (0.141), and `access_to_care_score` (0.134) rank
highest, followed by `housing_stability` (0.102) and `joint_pain_severity`
(0.095) — the split is driven by a genuine mix of clinical severity and
SDOH access features, not just one or the other. A Random Forest trained
to predict cluster label from the raw features gets 98% cross-validated
accuracy, confirming the clusters correspond to a real, learnable
pattern in the feature space rather than a K-Means artifact.

The two resulting groups:
- **Cluster 0 (mild / well-resourced):** low symptom severity, high
  housing stability, strong access to care/social support, short
  symptom duration, mostly privately insured, higher income.
- **Cluster 1 (elevated risk):** moderate-to-high symptom severity,
  longer duration, some disseminated disease, low housing stability, low
  access to care/support, mostly Medicaid/uninsured, lower income.

## 3. Evaluating cluster quality and usefulness

**Statistical validity:**
- **Silhouette / Davies-Bouldin / Calinski-Harabasz** — three internal
  metrics with different blind spots, used together rather than relying
  on one.
- **Cross-algorithm agreement (ARI)** — K-Means, Agglomerative, and GMM
  agree closely (ARI ≈ 0.92), so the structure isn't an artifact of one
  algorithm's assumptions.
- **Bootstrap stability** — mean ARI = 0.970 across 50 resamples; a
  grouping that shifted wildly on resampling would not be trustworthy
  enough to base care-coordination decisions on, however good its
  silhouette score looked on a single run.
- **Supervised recoverability** — a Random Forest predicts cluster
  label from the original features with 98% CV accuracy, confirming
  the split is a real, learnable pattern.

**Clinical/operational usefulness** — needs a domain expert, and matters
at least as much as the statistics:
- Do the profiles correspond to something a clinician or case worker
  would recognize and act on? (`profile_clusters()` outputs plain-
  language per-cluster summaries, not just labels.)
- Does group membership predict something we care about but didn't
  cluster on — hospitalization, time-to-diagnosis, treatment completion?
  That validation wasn't possible here (no outcome data), but it's the
  strongest real-world evidence of usefulness.
- Are groups actionable at the right granularity? A 2-patient cluster or
  a 90%-of-patients cluster isn't useful for resource planning.

**Equity/fairness audit** (`equity_audit.py`,
`outputs/equity_audit_categorical.csv`) — this is the check I'd expect
most take-home submissions to skip, and it turned up a real finding:
cluster membership is **strongly associated with income bracket**
(Cramér's V = 0.590), **employment status** (0.586), and **insurance
status** (0.519). That's expected, given SDOH features were included on
purpose — but it means this "risk group" is arguably as much a
*resource-access* group as a *clinical severity* group, and needs to be
communicated and used that way. If a grouping like this ever informed
resource allocation, the risk is using it to justify giving less care
intensity to patients whose access is already limited, rather than the
intended use of proactively directing outreach and support to them. That
tension is a policy/clinical-governance question, not something a
clustering script resolves — but a script that doesn't even surface it
is missing the most important real-world risk of this exact project.

## 4. From batch analysis to something usable: `assign_new_patient.py`

The rest of this pipeline clusters a static, historical dataset — useful
for understanding the population, but not directly usable by a case
worker with one new patient in front of them. `assign_new_patient.py`
fits the preprocessor + KMeans + GMM once on the historical population,
persists them (`joblib`), and scores new patients one at a time:

```bash
python assign_new_patient.py
```

For each patient it reports the hard KMeans cluster, the (properly
calibrated and label-aligned — see the bug writeup above) GMM
probability distribution over clusters, and a **boundary-case flag**: if
the top cluster probability is below 0.65, the tool recommends routing
the patient to a human case manager instead of auto-applying a group
care pathway. In this well-separated synthetic run, boundary cases are
rare on realistic patients — the exact-midpoint sanity check in the
script exists specifically to prove the flagging mechanism fires
correctly when it should, since that won't be true on noisier real data.

## 5. If I were extending this for a real deployment

- Replace synthetic data with real, de-identified, IRB-approved patient
  data; re-run k-selection and algorithm-agreement checks, since real
  data will have messier, non-Gaussian structure.
- Handle missing data explicitly — real SDOH survey data is often
  incomplete, and this pipeline currently assumes complete records.
- Validate clusters against downstream outcomes (hospitalization,
  time-to-treatment) if available.
- For Arizona specifically, Valley Fever is a reportable disease —
  county/state notifiable-disease surveillance data (Arizona Department
  of Health Services) would be the natural real clinical source, and
  CDC PLACES or the Area Deprivation Index could supplement or
  cross-validate self-reported SDOH fields.
- Consider a mixed-type distance measure (e.g., Gower distance) or
  k-prototypes as an alternative to one-hot + K-Means for high-
  cardinality categorical features.
- Run the equity audit on real data and bring the results to a domain
  expert / ethics review *before* any resource-allocation use, not
  after.
- Build a lightweight interface around `assign_new_patient.py` (rather
  than a CLI) so case workers can score patients without touching code.
