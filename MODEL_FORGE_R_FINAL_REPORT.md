# ModelForge-R / DocVQA Risk-Routing Study: Final Report

**Version:** publication candidate 1.0

**Closeout date:** 2026-08-09 (America/New_York)

**Scope:** frozen Phase A design, completed Phase B development study, post-hoc Phase C work, and terminal Phase D transaction

**Study status:** development evidence only; Phase D performance unmeasured

## Executive summary

This study asked whether signals available at DocVQA inference time could estimate the risk that a fixed Qwen3VL answer is an exact-match (EM) error or has ANLS equal to zero, allowing limited human review to be focused without changing the underlying answer. Phase A froze a document-grouped, leakage-controlled protocol. Phase B executed that protocol once on a 5,349-row baseline and remains development evidence with formal classification `PHASE_B_CLOSEOUT_GAP_FOUND`. Phase C added explicitly post-hoc reporting and development analysis; it is not confirmatory and cannot substitute for Phase D.

On the frozen Phase B test partition, the combined EM-error router had materially better probability scoring than confidence-only. Its Brier score was lower and the paired grouped-bootstrap Brier interval excluded zero. Ranking and selective-review improvements were smaller and uncertain. The benefit did not generalize reliably to ANLS zero: confidence-only was more effective for prioritization, and isotonic score ties made the scalar-threshold policy automate about 99.13% rather than approximately 80%.

Phase D did not produce a model-performance result:

> Phase D label-blind inference was completed and predictions were sealed. However, the one-shot confirmatory evaluation terminated after the irreversible start boundary because at least one selected row lacked a valid reference answer. No scientific metrics were produced. Phase D performance therefore remains not estimable, and the confirmatory claim was not established.

The terminal classification is `PHASE_D_ONE_SHOT_EVALUATION_STARTED_TERMINAL_FAILURE_NO_RETRY`. This outcome does not establish superiority, equivalence, or inferiority.

## Problem definition and intended use

The framework estimates endpoint-specific error probabilities for already-produced document answers. The intended use is research on selective human review under a fixed feature and provenance contract. The router does not replace the VLM, repair answers, prove correctness, or establish a safety guarantee.

Prohibited or unsupported uses include autonomous acceptance or rejection, safety-critical decisions, production deployment based on this study, official full-benchmark claims, cross-domain generalization, tuning on test or holdout labels, use of question types or ground truth as production features, use of the rejected prompt candidate, and interpretation of a risk probability as a correctness guarantee.

## System architecture and method

The retained ModelForge-R package supplies contracts, deterministic artifact handling, local backends, feature construction, model/calibrator factories, grouped evaluation, monitoring, qualification policies, and reporting. The DocVQA example layers a frozen endpoint-specific risk-routing experiment on those governance ideas.

The Phase B production features were confidence, clipped confidence logit, generated-answer character/word/line/token lengths, question character/word lengths, question-text-only expected-answer form, and two predefined confidence-length interactions. Answers, correctness, EM, ANLS, question types, target-derived fields, and calibration/test-derived fields were prohibited. Question types were allowed only for reporting. The rejected prompt candidate was excluded from every input and selection path.

For EM error, selection chose histogram gradient boosting with maximum leaves `7` and Platt calibration. For ANLS zero, selection chose logistic regression with `C=0.1` and isotonic calibration. Model family, features, interactions, hyperparameters, and calibrator were selected using training data. The threshold and operating policy were selected using calibration data. Test labels were opened only after every selection decision was frozen.

## Data, terms, and leakage controls

Phase B used 5,349 questions from 938 UCSF document groups. The persisted split was:

| Role | Rows | Documents |
|---|---:|---:|
| Train | 3,168 | 563 |
| Calibration | 1,143 | 188 |
| Test | 1,038 | 187 |

Document sets were mutually disjoint. Stable question identities controlled joins and deterministic tie-breaking. Bootstrap resampling used documents rather than rows.

The later Phase D population was a project-specific, document-disjoint derivative subset of 3,230 rows, 718 documents, and 786 pages. It was not the official full DocVQA test benchmark. No dataset is distributed here. Data must be obtained from the [official DocVQA site](https://site.docvqa.org/datasets/docvqa) and its [RRC route](https://rrc.cvc.uab.es/?ch=17&com=tasks), subject to the applicable terms. Publisher metadata for a formatted derivative does not conclusively override underlying RRC or UCSF conditions.

## Experimental phases

### Phase A: frozen offline design

Phase A implemented deterministic UCSF-document-grouped 60/20/20 splitting, stable identities, a production-feature allowlist and prohibited-field denylist, train-only candidate selection, calibration-only thresholding, one-shot test access, exact candidate-grid enforcement, undeclared-candidate rejection, lexical tie-breaking, separate EM-error and ANLS-zero endpoints, confidence-only and constant baselines, and 2,000 document-grouped bootstrap replicates with seed `20260808`. Unit and synthetic end-to-end tests covered these boundaries. Phase A stopped before real-data fitting or evaluation.

### Phase B: one-shot development evidence

The authorized Phase B command bound the baseline prediction hash and the source-manifest hash literally, used seed `20260808`, and requested 2,000 grouped bootstrap replicates. STARTED and COMPLETED markers belong to the same non-overwriting transaction. No second marker pair, retry, residual staging transaction, or overwrite was evidenced. This is strong persisted evidence of one execution, though not a hardware-backed invocation counter.

Phase B remains formally `PHASE_B_CLOSEOUT_GAP_FOUND` because three reporting or traceability items were not persisted: question-type subgroup summaries and uncertainty; a literal seed field inside each bootstrap result object; and calibration-split Brier, ECE, and reliability diagnostics. These gaps do not rewrite the completed core transaction, but they constrain its closeout.

### Phase C: post-hoc exploratory evidence

All Phase C findings are `POST_HOC_EXPLORATORY_PHASE_C_NOT_PHASE_B_CONFIRMATORY`. Phase C produced question-type summaries with broad uncertainty and small/class-limited groups; no subgroup superiority claim is supported. It also computed retrospective calibration-split diagnostics on data used to fit/select calibration. Those values are resubstitution descriptions, not independent validation.

Development-only work retained the EM HGB/Platt candidate. For ANLS zero, deterministic exact-count capacity control achieved the intended batch coverage mechanically, unlike the scalar threshold. This is capacity accounting, not evidence of improved ranking, calibration, or generalization. No improved candidate was scored on the frozen Phase B test set.

### Phase D: sealed inference and terminal evaluation refusal

Label-blind inference completed once with zero inference retries and sealed predictions for 3,230 rows. The durable evaluation STARTED boundary was then crossed and the canonical evaluator was invoked exactly once. It terminated with `PhaseDRefusal: selected row has no valid reference answer`. Metric computations were zero, bootstrap invocations were zero, and evaluation retry/resume counts were zero. No completion marker, result, partial result, or final scientific output exists. The scalar answer/reference-access count is `UNKNOWN`; one post-START projection-construction event is separately recorded.

The original one-shot opportunity is consumed. Retry, resume, answer/subset repair, evaluator patching, manifest requalification as revival, or reuse of the sealed predictions for a corrected original-Phase-D claim is prohibited. A future study would need a new protocol and must not be described as Phase D confirmation or retry.

## Results

### Phase B EM-error endpoint — development evidence

The selected threshold was `0.28033061947034527`. Calibration coverage was `0.799650043744532`; frozen-test coverage was `0.8111753371868978` (842 automated questions).

| Method | Brier | ECE | AUROC | AUPRC | AURC |
|---|---:|---:|---:|---:|---:|
| Constant | 0.13086148930645425 | 0.021882480878145594 | 0.5 | 0.15414258188824662 | 0.1601841755645305 |
| Confidence-only | 0.13955756616578271 | 0.13237206344308705 | 0.7762314920273348 | 0.3718113252029318 | 0.06137411598447162 |
| Combined | 0.1107033370336411 | 0.042997220936498076 | 0.7911980353075172 | 0.42410588633681123 | 0.0595392584395116 |

Paired combined-minus-confidence grouped-bootstrap intervals were: Brier `[-0.040160771887003247, -0.01762336311735728]`, ECE `[-0.11299570802732664, -0.05049539561345925]`, AUROC `[-0.00498693525854389, 0.03406779606185204]`, AUPRC `[-0.010358491081650857, 0.10716204526363048]`, and AURC `[-0.006977006366621793, 0.0041462334541349025]`. At 5% review, combined captured 30/160 errors and confidence-only 21/160; the capture-difference interval was `[0, 0.10417687908496728]`.

The combined router therefore improved probability scoring in this development analysis. Ranking and selective-review benefits remain uncertain. Brier reflects calibration, resolution, and outcome uncertainty; a lower Brier score alone is not proof of calibration.

### Phase B ANLS-zero endpoint — development evidence

The scalar threshold was `0.1375`. Calibration coverage was `0.9746281714785652`; frozen-test coverage was `0.9913294797687862` (1,029 automated questions), far from the intended approximately 80%.

| Method | Brier | ECE | AUROC | AUPRC | AURC |
|---|---:|---:|---:|---:|---:|
| Constant | 0.046718943910296094 | 0.00020618030011092248 | 0.5 | 0.049132947976878616 | 0.05431544826979888 |
| Confidence-only | 0.04365280560382604 | 0.027714583205627828 | 0.773168842004887 | 0.22446146797524405 | 0.019887254556541174 |
| Combined | 0.043486576527501535 | 0.018323578202198543 | 0.748932196992272 | 0.18591500062424077 | 0.032529385653072125 |

Every paired interval included zero: Brier `[-0.0019232328854048605, 0.0013674916502144005]`, ECE `[-0.02522048475712615, 0.003358631314501884]`, AUROC `[-0.04813335566757983, 0.0010452030585718435]`, AUPRC `[-0.11573760840502215, 0.019892441034011322]`, and AURC `[-0.002342518846286109, 0.030397150713225154]`. At 5% review, confidence-only captured 12 errors and combined 9; the capture-rate-difference interval was `[-0.15555555555555553, 0.052655075187969795]`. The combined ANLS-zero router did not reliably improve and was operationally unsuitable under the scalar-threshold policy.

### Phase C diagnostics — exploratory only

Retrospective calibration-split values were EM combined Brier/ECE `0.120732`/`0.042880` versus confidence-only `0.159650`/`0.150770`; achieved coverage was `0.799650`. ANLS isotonic Brier was `0.047349`, ECE was effectively zero, and scalar-threshold coverage was `0.974628`. These reuse calibration data and are not independent estimates.

### Phase D — no result

| Evidence | Status |
|---|---|
| Label-blind inference | Completed; predictions sealed |
| Evaluation transaction | STARTED; canonical evaluator invoked once |
| Terminal event | `PhaseDRefusal: selected row has no valid reference answer` |
| Scientific metrics | None produced |
| Bootstrap runs | 0 |
| Retry/resume | 0 |
| Performance | `UNEVALUATED / NOT ESTIMABLE` |
| Confirmatory claim | `NOT CONFIRMED` |
| Directional comparison | Not established |

There is no numeric Phase D result.

## Limitations and failure modes

- Phase B is development evidence, not independent confirmation.
- Phase C subgroup and calibration analyses are post-hoc; resubstitution and multiplicity limit inference.
- Isotonic score ties make scalar-threshold coverage discontinuous and caused near-total ANLS-zero automation.
- Exact-count capacity control can meet a review count while ranking poorly, requires a declared batch, and has no canonical row-by-row streaming meaning.
- Low-prevalence endpoints and small subgroups make estimates unstable.
- Stable-ID absence or duplication breaks deterministic joins and tie resolution.
- Confidence and combined models may shift across document domains.
- The Phase D population was project-specific, not the official benchmark.
- Phase D reference eligibility failed after STARTED; performance is not estimable.
- The expected external label-blind manifest anchor `a936a9cbccc326b5fa4f31560289aa6d0013b8385425cf20d0f8ff73c3e01705` differs from the observed manifest `b84a9a9a79ec207f820a285f593e13e97de0513e75d11e7039d8beb872cf9776`; the discrepancy remains unresolved.
- Byte hashes prove present-byte identity, not scientific truth, lawful acquisition, or historical non-use.

## Reproducibility

This publication candidate preserves the portable ModelForge-R source, configurations, dependency declarations, offline-safe tests, and the DocVQA risk-routing source/specification/tests. It intentionally does not include the immutable baseline predictions, model artifacts, sealed Phase D predictions, datasets, or transaction outputs. Consequently, it supports source inspection and synthetic/offline validation, not recreation or retry of the protected transactions.

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src examples/docvqa_risk_routing/scripts
python -m pytest -q tests
python -m unittest discover -s examples/docvqa_risk_routing/tests -v
python -m ruff check src tests examples/docvqa_risk_routing/scripts examples/docvqa_risk_routing/tests
python -m ruff format --check src tests examples/docvqa_risk_routing/scripts examples/docvqa_risk_routing/tests
```

Do not treat these checks as a scientific rerun. Real inputs and outputs are protected and not distributed.

## Public-safe artifact and provenance ledger

### Authoritative closeout documents

| Artifact | SHA-256 |
|---|---|
| `DOCVQA_RISK_ROUTING_INTERIM_REPORT.md` | `a0b091fcb964fdc674ee1be087da50d20fb8c6b061f596a5566188b52cdcd0a4` |
| `PHASE_B_PROVENANCE_LEDGER.md` | `8322e48796c4191a2f03c7e637c3b684d8cb3a370d3245d3219033cc08d1ad2b` |
| `PHASE_B_MODEL_CARD.md` | `6620cf28d1c3a1568beb5d8ff9de787ea7dc32d8b2dcfb6ea03023e4a712671c` |

### Phase B

| Artifact or pin | SHA-256 |
|---|---|
| Phase B source manifest | `6895ddeb6a9059f76cb55b3554eab41e0d0cab52789effd58214c52071bec142` |
| Approved baseline predictions | `cf0157f4b52b330e2d9f024fd74267c44db1a2a1cf15602f87d435912eef7efb` |
| Rejected candidate predictions | `32a69e7c75e0a44bbe110e3294303031216daea23a87274c5a1ec098b5cd671b` |
| Prompt comparison / sidecar | `1d44f3d32a2b079186851ac4bcf3186835bbf533aab6c9f0cf7703f344e3b1fb` / `fd1817b4425ac96d61b5b62590fa6a7bec14f14c142e8cd7346031fa254bd621` |
| Rejection record / sidecar | `e12681ca55b3ec93bcaee1d01d8bc6ac9cb7d6f70cdec3e2208da49878944bc0` / `5709b013fe2a9273f46932f00c8186e4cabdde2294852677d6dfb7901189de3c` |
| EM router | `19abaedb2b4a7ca0466b7b86822d2c07b01b988c7ed8c58d50a3183ae19fa682` |
| ANLS-zero router | `28d80d3902e3843ca14cb9a6925339c4612bf84cf0770c7c559fe6c7d4cdbe49` |
| Phase B evaluation | `c864709e25e2258e723669d9d60d9e5401e25cae6f770b711619cab57e112d01` |
| Phase B STARTED / COMPLETED | `4f64d7ffb37a5d6845812de6b5372ed55b6d1029c10287640c147a3238ded52a` / `57e6ad5c29f490ac517715e6d0d19355d2488e65087bf47ce58a2fe0c93bffb9` |

### Phase C and continuation

| Artifact | SHA-256 |
|---|---|
| Phase C reporting | `b71051432724228fcb4b460081dc377e64fb8a88f40fb002bb51f98f0ae6060d` |
| Phase C improvement | `064d1b4a022be46ab0894aa3986745c906c1f50457ac70652e0746a924d1260f` |
| Completed continuation manifest | `6d04c73542b096397f31673173e8a3ba5789d3bd6d98f9e541a3c6fe3c3ca68b` |

### Phase D controlling audit

| Artifact | SHA-256 |
|---|---|
| Transaction decision | `096d5d51b02e062010aa3a81c3995815fec6c5a63f0c33cf38b88686a0eacdc5` |
| Transaction report | `75f8fdc0d6c6c1dc9c0153109770d47118458ead7eb14d15aee8aaffd25cfe52` |
| Transaction inventory | `9426c9e69759ef40a62668633f5ff18333a4b72fd45d3b3b9307e8476e209472` |
| Transaction checksums | `5e423bfe37620c4f2f4a3ea7573db269760cfd08ec89d7f0ffbe18a449a48fbf` |
| Sealed predictions | `1cc021e42b05eef3c00ec20096eb1b0aa3d853f10c7ff170b1a18e10aa6fcb78` |
| Durable Phase D STARTED marker | `6286bcc70e287e65e7cca791b634377fb0031cab53a118a1749ba783584f68b5` |

The documentation-closeout package is anchored by decision `85b8597a9477de454ff540b19c96c1bb0e6f6d0c54d1d381b135f6218ae9dd5c`, report `ac644d217a066622496284dd076e13e36f2a64a8f0894b1c84824cf1a92b6751`, inventory `acd028a2cf7af61bd9844b7b581f2e084e422e354458c05538a35f0962d5150d`, and checksums `e443b2a9701d20a35f6c951f264826a288896c9b4e88affe2ddcd907fda75f8d`.

Protected artifacts and datasets are not distributed.

## License, citation, acknowledgments, and contact

ModelForge-R source is provided under the repository's MIT license. Dataset rights and terms are separate and must be obtained from the respective publishers. The DocVQA study draws on the DocVQA task and UCSF-sourced document imagery described by the official dataset materials; no dataset license is granted by this code repository.

For citation, cite this repository/version and the original DocVQA publication identified by the official project page. The package metadata identifies Jinwoo Bae as the framework author. A public contact channel may be added by the repository owner before publication; none is invented here.

## Final claim boundary

Phase B is bounded development evidence with unresolved closeout gaps. Phase C is post-hoc and exploratory. Phase D is unmeasured and terminal. This study does not authorize deployment and does not establish a confirmatory directional comparison.
