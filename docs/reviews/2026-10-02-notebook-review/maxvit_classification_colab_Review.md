# MaxViT-Tiny Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/maxvit-classification-pipeline`  
**Notebook:** `tutorials/maxvit_classification_colab.ipynb`  
**Reviewed commit:** `42ffa57187f1d974926e69713f7e4d3b5fe6e141` (`main`, confirmed with `gh api repos/kurtvalcorza/maxvit-classification-pipeline/commits/main`)  
**Notebook Git blob:** `b736baab41ad20b8936942a134124e6a8bd25e49`, the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-26 (commit `0c1164e`). The notebook last changed in `2af137c`.  
**Finding prefix:** `MXV`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.1.

## Executive assessment

The default path is carefully built. It does the following:

- Carries `data.py` and `pipeline.py` verbatim and asserts the inline manifest against the module identity.
- Stages and re-hashes the pinned `timm/maxvit_tiny_tf_224.in1k` snapshot.
- Downloads a digest-pinned 400-image CIFAR-10 subset (`frog`, `truck`).
- Measures majority-class, zero-shot ImageNet-mapping and untrained-head baselines before a bounded full fine-tune.
- Evaluates on a held-out split and an unseen split.
- Probes blank and noise images before and after adaptation.
- Exports a SafeTensors adapter and checks that it reloads onto a fresh base within a stated tolerance.

Its prose on softmax scores, the missing reject option and the small-sample limit is correct. This notebook comes from the same generator family as the ConvNeXt V2, CvT and EfficientNet notebooks (relay rows 15, 17 and 23). It shares their duplicate-image split leak and their `FREEZE_BACKBONE` rerun defect, and this review reproduced both here.

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (MXV-M1).** The recorded Kaggle run stopped at the install cell's stale-module guard and passed only after a restart. The release record still reports it as PASSED.
2. **The evaluation splits contain exact copies of training images, while Section 5 says the images are independent (MXV-M2).** At the default scale, 25 of 60 held-out images and 19 of 60 unseen images are pixel copies of a training image.
3. **The documented head-only experiment corrupts the run (MXV-M3).** Re-running cell 19 with `FREEZE_BACKBONE = True` trains a head on the already fine-tuned backbone. Cell 25 then raises an `AssertionError`.
4. **The comparison is at ceiling, and the prose presupposes a gain (MXV-M4).** The zero-shot ImageNet mapping already scores 1.000 on held-out, equal to the fine-tuned head (hosted record and this review). Yet the notebook says the baselines give the fine-tune "something to beat" and that "the fine-tune … is what moves the score". "Try next" also asks the learner to watch the fine-tune "fall back toward" a baseline that already scores 1.000.
5. **The guided layer is largely absent (MXV-M5).**

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.1** |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: basic Python and PIL; softmax; convolution and self-attention "at the level of local versus global context"; accuracy, balanced accuracy, confusion matrix; why a baseline is needed |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA T4 documented, CPU also runs |
| Promised outcomes | Pinned install; carried modules; digest-verified snapshot and sample; validation before any model; ImageNet top-5 + `sample-sanity`; zero-shot baseline; blank/noise probes; stratified split; untrained-head floor; bounded fine-tune; held-out comparison; unseen inference; adapter export + verified reload; five `outputs/` files; BYOD image and dataset branches ("the same validate → split → baselines → fine-tune → evaluate → export → reload stages as the sample"); "Try next" experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; generating revision `6b59382` |
| Release status | `Candidate` (`tutorials/README.md`, `STATUS.md`, `docs/release-verification.md`); REL12 pending |

### Evidence actually obtained

- **Source inspection.** I read the following:
  - all 31 cells (14 code; cells 5 and 7 are the carried `data.py`, 341 lines, and `pipeline.py`, 818 lines);
  - the generator and template;
  - `data.py` (`validate_dataset`, `split_dataset`) and `pipeline.py` (`finetune`, `save_artifact`, `load_artifact`);
  - `README.md`, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md`.

  The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Documented execution evidence.** `docs/release-verification.md` records a Kaggle Tesla T4 run of **the reviewed blob** on 2026-09-26:
  - pass 1 stopped with a `RuntimeError` in the install cell; pass 2 completed 14/14 after a restart;
  - held-out accuracy: majority 0.500, zero-shot 1.000, untrained head 0.3667, fine-tuned 1.000; unseen 1.000;
  - adapted head: blank image → `frog` 0.828, noise → `truck` 0.756.

  No Colab run, BYOD dataset run or experiment run is recorded.
- **Direct execution (this review).** All runs used `run_probes.py` on Windows 11, CPU only, with Python 3.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, timm 1.0.29, safetensors 0.8.0, huggingface-hub 0.36.2, numpy 2.5.3 and pillow 11.3.0. These are the notebook's pins (CPU wheel), taken read-only from another pipeline repository's `.venv`; nothing was installed. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so the restart behaviour rests on the documented run only. Each probe started from an empty working directory and an empty `HF_HOME`, and the snapshot and archive were fetched and digest-verified.
  - **P0 (static checks):** markers, install pattern, `cellView`.
  - **P2F, full scale (`PER_CLASS = 200`, every field at its default, cells 3–13, no fine-tune):** the split, pixel-digest overlap between train and the evaluation splits, and zero-shot on the copy-free held-out subset.
  - **P1 at REDUCED SCALE:** every code cell with `PER_CLASS = 30` substituted for 200 (all other fields default). A full-scale CPU fine-tune of MaxViT-Tiny did not fit the probe time cap. These numbers are a stand-in, not a reproduction of the hosted record.
  - **P3 at reduced scale:** after P1, `FREEZE_BACKBONE = True` and a rerun of cells 19, 21, 23 and 25.
  - **P4:** the BYOD cell with a compatible 12-image class-folder directory, a single image, and five incompatible inputs (path fields only, no upload dialog).
- **Not verified:** a full-scale direct fine-tune (the hosted record covers it); a Colab run; the Colab upload dialog; the `PER_CLASS` experiment; GPU timing.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness:** strong on the default path. Hosted: 14/14. Reduced direct: 14/14 in 79 s, with the reload equivalence check passing and five outputs written. There are two defects:
  - the install pattern forces a restart (MXV-M1);
  - `finetune` mutates the pipeline it is called on, so the documented experiment fine-tunes an already fine-tuned model and breaks the reload check (MXV-M3).
- **Scientific validity:** the split ignores the archive's pixel-identical pairs (MXV-M2). The task is also already solved by the zero-shot mapping, so the held-out comparison cannot show what fine-tuning adds (MXV-M4).
- **Promise fulfilment:** the default promises are met. The head-only experiment is not delivered (MXV-M3). The BYOD dataset reload is not checked the way the sample's reload is (MXV-m1).
- **Learner experience:** the prose is accurate and candid, but the framing assumes a fine-tuning gain the run does not show (MXV-M4), and the guided layer is missing (MXV-M5).
- **Spec conformance:**
  - Unresolved applicable MUSTs: RUN1, RUN10, ENV6 and REL2 (MXV-M1); SPL5 (MXV-M2); REL12 BYOD evidence absent from the release record.
  - SHOULD deviations: SPL10; GDL10, UX5 and VER4 (MXV-M3); EVAL10 and GDL14 (MXV-M4); GDL1–GDL7, GDL9–GDL14 and UX8 (MXV-M5); DAT19 and UX10 (MXV-m2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 14/14 | Section 1 says the cell "stops with a restart instruction" | **Not met** (MXV-M1) |
| Digest-verified snapshot and sample, validated before any model | cells 9, 11 | 3/3 files verified; 400 records accepted with "100 group(s) of pixel-identical images" | finding printed, never discussed | Met; finding unexplained (MXV-M2) |
| Stratified split, valid held-out | cell 13 | 280/60/60, ID-disjoint; 25/60 held-out and 19/60 unseen are pixel copies of training images (P2F) | "valid here because CIFAR-10 images are independent thumbnails" | **Not met** (MXV-M2) |
| ImageNet top-5 + zero-shot baseline | cell 13 | frogs → `tailed frog` 0.177 / `ocarina` 0.302; zero-shot held-out **1.000** | grouped-mass rule explained | Met |
| Fine-tune beats baselines ("something to beat") | cells 17–21 | fine-tuned 1.000 = zero-shot 1.000 (hosted; reduced direct likewise) | prose presupposes a gain; tie never discussed | **Not met as framed** (MXV-M4) |
| Blank/noise probes | cells 15, 23 | adapted head picks a confident class for structure-free input | explained | Met |
| Unseen inference | cell 23 | 1.000 | "never used for training" — 19/60 are copies | Met, overstated (MXV-M2) |
| Adapter export and verified reload | cell 25 | 10-image (reduced) equivalence within 1e-4 | "loading succeeding is not the check" | Met |
| "Try next: `FREEZE_BACKBONE = True` and compare … scores and runtime" | cell 19 field | P3: head trained on the fine-tuned backbone; cell 25 `AssertionError`; no runtime printed | none | **Not met** (MXV-M3) |
| BYOD dataset through the same stages | cell 29 | P4: 12-image directory validated, split, fine-tuned, exported, reloaded; no reload comparison, no unseen split | contract stated first | Partly met (MXV-m1) |
| BYOD image | cell 29 | P4: top-5 + `not-measurable` | stated | Met |

The learning objectives are operations the code performs (GDL5). The only learner-controlled activity is "Try next", which breaks the run (MXV-M3) and presupposes its outcome (MXV-M4).

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | Each stage is introduced; two "What to look for" notes. Missing: audience, how-to-use, roadmap, task contract, glossary (MBConv, block/grid attention, softmax mass, balanced accuracy, frozen backbone, adapter), prediction prompts, checkpoints, troubleshooting, conclusion template. The 1,159 carried lines are unlabelled (MXV-M5). The duplicate finding is never explained and Section 5 asserts the opposite (MXV-M2). With zero-shot at 1.000 a learner has no guidance for reading a tie (MXV-M4). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped; full-scale split cells 3–13; all cells at **reduced** `PER_CLASS = 30`) | Kaggle: restart needed after pass 1 (MXV-M1), then 14/14. Full-scale split: 280/60/60, zero-shot held-out 1.000 / 1.000, mean group mass 0.4214, top-5 sample predictions identical to the hosted record. Reduced: 14/14 in 79.4 s (fine-tune 63.7 s); held-out (8) majority 0.500, zero-shot 1.000, untrained 0.125, fine-tuned 1.000; unseen (10) 1.000; losses 0.630 → 0.040; reload equivalent; five outputs. No Colab run. |
| **Active learning** | Direct, reduced scale (P3) | After the reduced default run: `FREEZE_BACKBONE = True`, rerun of cells 19 → 21 → 23 → 25. Cell 19 reports 264,706 of 30,404,554 trainable and calls the run frozen, but the stem weights are the fine-tuned ones (`stem_weight_equals_imagenet_base: false`). The first epoch loss is 0.0927 (fresh default: 0.6296). Cell 21 shows fine-tuned 1.000 again. Cell 25 raises `AssertionError` at the score-tolerance assertion. `PER_CLASS` experiment not run. |
| **Reuse and recovery** | Direct (P4, path fields) + source; Colab upload dialog not verified | Compatible 12-image directory: accepted, split, untrained 0.25 / fine-tuned 1.00 balanced accuracy, adapter exported and reloaded (22 s), no equivalence check. Single image: top-5 (`tailed frog` 0.241), `not-measurable`. Incompatible: one-image class → "every class needs at least 2 images to split; short classes: {'b': 1}"; non-image member → "b\2.png: not a decodable image (…)"; missing directory → `FileNotFoundError … is not a directory`; text file to the image branch → raw `PIL.UnidentifiedImageError` (MXV-m2). |

## 5. Findings

### Major

#### MXV-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1. Generator: `tools/build_notebook.py` `_INSTALL_GUARD` (lines 50–72) and the install-cell assembly (line 469). Also `docs/release-verification.md` Recorded executions, plus the status lines in `STATUS.md` and `tutorials/README.md`.
- **Observed issue:** the cell `pip install`s seven pins into the running kernel. When a loaded distribution changed, it then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` The opening cell promises that **Run all** in a fresh runtime completes every stage.
- **Consequence:** on a stock Kaggle or Colab image, the first code cell errors and the learner must restart and run again. "PASSED (default path)" rests on that restart-dependent run.
- **Evidence:** documented: the Kaggle T4 run of blob `b736baab`; pass 1 stopped (`cuda-bindings 12.9.4 → 13.4.3`, `numpy 2.0.2 → 2.5.3`); pass 2 was 14/14 "post-restart". Source: P0 `pip_install_in_kernel: true`, `uses_uv: false`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**:
  1. the setup cell bootstraps uv;
  2. it creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`;
  3. it installs a hash-locked `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`;
  4. it runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced.

  References: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` on `main`. Implement it in `tools/build_notebook.py`, regenerate, and re-qualify with a one-pass hosted run.
- **Acceptance check:** this test passes when both of the following hold.
  - A fresh Kaggle or Colab runtime completes every code cell in one pass with no restart, recorded with the blob id and `restarted: false`.
  - `grep -n "Restart the runtime" tutorials/maxvit_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### MXV-M2 — The evaluation splits contain exact copies of training images; Section 5 says the images are independent

- **Cell/section:**
  - cell 12 (Section 5 prose; `tools/notebook_template.py` line 154);
  - cell 13 (split with an ID-only leakage assertion);
  - cell 22 ("never used for training");
  - cell 11 (duplicate finding printed, never discussed);
  - `data.py` `split_dataset`.
- **Observed issue:** the archive stores each thumbnail under `original_images/` and `darkened_images/`, and 100 of the 200 pairs in the sample are pixel-identical. `split_dataset` splits per record, and the assertion compares IDs, which differ between the two folders. Yet `validate_dataset`'s own docstring notes that "a duplicate inflates the held-out score when one copy lands on each side of a split". The notebook says: "A random split is valid here because CIFAR-10 images are independent thumbnails."
- **Consequence:** at the defaults, 25 of 60 held-out and 19 of 60 unseen images are exact copies of a training image. The fine-tuned row and the "unseen" score are therefore partly memorisation checks, and the learner is taught a false reason for the split.
- **Evidence:** direct, full scale (P2F): `pairs_pixel_identical` 100 of 200; held-out exact copies 25/60 (counterpart in train 46/60); unseen 19/60 (36/60). The copy-free held-out subset is 35 images (30 frog, 5 truck), on which zero-shot scores 1.000. The full-scale fine-tuned score on that subset is not measured (no full-scale CPU fine-tune). Documented: `docs/release-verification.md` and `STATUS.md` already record the 46/60 counterpart overlap and that the split is not duplicate-aware. Same defect as rows 15, 17 and 23.
- **Recommended correction:**
  - Split by source image: keep each `original_images`/`darkened_images` pair (or each pixel-digest group) on one side. Alternatively, read only `original_images/`.
  - Add a pixel-digest disjointness assertion.
  - Replace the "independent thumbnails" sentence.
  - Have cell 11's prose interpret the duplicate finding.
  - Re-record the comparison afterwards.
- **Acceptance check:** no held-out or unseen image's pixel digest equals any training image's digest (an in-notebook assertion passes). No pair is split across sides. Section 5 no longer claims the archive's images are independent.
- **Spec:** SPL5 (MUST), SPL3, SPL10.

#### MXV-M3 — The documented "head-only" experiment fine-tunes the already fine-tuned model and breaks the reload check

- **Cell/section:**
  - Interpretation "Try next" (`tools/notebook_template.py` lines 458–459);
  - cell 19 (the `FREEZE_BACKBONE` field and `adapter.finetune`);
  - cell 25 (reload assertions);
  - `pipeline.py` `finetune` ("mutating this pipeline") and `save_artifact` (writes only non-frozen tensors).
- **Observed issue:** no rerun scope is given. Re-running cell 19 calls `finetune` on the same `adapter` that the default run already fully fine-tuned. `save_artifact` then writes the head only, and `load_artifact` overlays it on the pristine ImageNet backbone, which gives a different model.
- **Consequence:** the only documented experiment reports a frozen fine-tune that is not one. Its held-out 1.000 is inherited from the full fine-tune, and the run ends in an unexplained `AssertionError`. Any head-only versus full conclusion drawn from it is wrong.
- **Evidence:** direct, reduced scale (P3): `freeze_backbone: true`; 264,706 of 30,404,554 parameters trainable; first-epoch loss 0.0927 against 0.6296 for a fresh run; stem weights not equal to the ImageNet base; cell 25 `AssertionError` (line 14, score tolerance). Same defect as rows 15, 17 and 23.
- **Recommended correction:** give the experiment a fresh re-headed base. Either move the field into cell 17 with "re-run cells 17–27", or create the adapter inside cell 19 before `finetune`. Show the frozen row beside the default row. State the rerun range for `PER_CLASS` too (cells 11–27).
- **Acceptance check:** after a default Run all, following "Try next" exactly yields a frozen run that meets all three conditions:
  - its backbone equals the ImageNet base after training;
  - its own held-out row appears beside the full fine-tune row;
  - cell 25 passes.
- **Spec:** GDL10, UX5, VER4, FT5.

#### MXV-M4 — The task is at ceiling for the zero-shot baseline, but the notebook frames the fine-tune as the thing that moves the score

- **Cell/section:**
  - cell 12 ("Two baselines give the fine-tune something to beat", template line 156);
  - cell 16 ("the fine-tune — not the re-heading — is what moves the score", line 214);
  - cell 20 / Section 9 (no reading of a tie);
  - Interpretation "Try next" ("watch how quickly the fine-tuned model falls back toward the zero-shot baseline", line 459).
- **Observed issue:** with this checkpoint, the zero-shot ImageNet mapping already scores 1.000 accuracy and balanced accuracy on the held-out split. The fine-tuned head also scores 1.000. The comparison therefore cannot show any benefit of adaptation. The notebook never says so: it frames the baselines as targets the fine-tune beats, and its "Try next" assumes a gap between the fine-tuned model and the zero-shot baseline.
- **Consequence:** the learner is likely to conclude that fine-tuning improved the model, or to look for a difference that isn't there. The `PER_CLASS` experiment as worded cannot be observed, because the fine-tuned model cannot "fall back toward" a baseline that sits at 1.000. The repository's own record states that "this run shows no gain from fine-tuning", but the notebook does not.
- **Evidence:** documented: Kaggle record, zero-shot 1.000 = fine-tuned 1.000 on 60 held-out. Direct: full-scale P2F zero-shot 1.000 on 60 held-out and 1.000 on the 35 copy-free images; reduced P1 zero-shot 1.000 = fine-tuned 1.000. Source: the quoted prose.
- **Recommended correction:** choose one:
  - (a) choose a class pair, or a dataset slice, on which the zero-shot mapping is measurably below ceiling on a duplicate-aware split, and record it; or
  - (b) keep the pair, make Section 9 explain what equal scores at ceiling mean (adaptation adds nothing measurable here; the comparison checks that the new head learned the task, not that it is better), and remove the presupposed gap from cells 12/16 and "Try next".

  Either way, phrase the experiment as a prediction to test.
- **Acceptance check:** this test passes when either of the following holds.
  - A recorded hosted run shows zero-shot held-out balanced accuracy below 1.0 on a duplicate-aware split.
  - The notebook text contains an explicit reading of a zero-shot/fine-tuned tie, and no sentence asserts or presupposes that the fine-tune beats or sits above the zero-shot baseline.
- **Spec:** EVAL10, GDL14 (framework dimensions 3 and 5).

#### MXV-M5 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5 and 7, and the end of the notebook. Generator: `tools/notebook_template.py` and the section assembly in `tools/build_notebook.py`.
- **Observed issue:**
  - Missing: an intended-learner statement, **How to use this notebook**, a roadmap, an Input → Model → Output task contract, a glossary, predictions before the baselines/fine-tune/comparison, interpretation checkpoints with sample answers, troubleshooting (Hub download, digest mismatch, CPU time, memory, BYOD) and a conclusion template.
  - Cells 3, 5 and 7 (52 + 341 + 818 lines) carry no **Infrastructure** label and no `cellView: form`.
- **Consequence:** a self-paced learner gets accurate prose but little help in deciding what matters or stating a conclusion, and the carried code dominates the scroll.
- **Evidence:** source; P0 `guided_markers`: how_to_use, roadmap, glossary, audience, predict_prompt, troubleshooting, conclusion_template and infrastructure_label are all false. The `checkpoint` match is the word for model weights. `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template, following NOTEBOOK_SPEC 2.2. Build the GDL10 activity on the MXV-M3/MXV-M4 fixes. Title cells 3, 5 and 7 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL7 and GDL9–GDL14 maps to a named cell in a checklist in `tutorials/README.md`, and cells 3, 5 and 7 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL7, GDL9–GDL14, UX8.

### Minor

#### MXV-m1 — The BYOD dataset branch does not run "the same stages as the sample"

- **Cell/section:** cell 28 prose, cell 29 dataset branch (`tools/notebook_template.py` around lines 386–440).
- **Observed issue:** compared with the sample path, the branch:
  - uses a two-way split (no unseen split);
  - prints balanced accuracy only;
  - runs no new-data inference;
  - calls `load_artifact` without comparing predictions, whereas the sample checks equivalence within a tolerance;
  - runs no duplicate check across its split.
- **Consequence:** a user's adapter is "exported and reloaded" without evidence that the reload reproduces the model.
- **Evidence:** direct (P4): the 12-image directory ran to "BYOD adapter exported and reloaded: 4ad2540dda423d63" in 22 s with no comparison.
- **Recommended correction:** reuse the sample cells' stages, including the tolerance check, or narrow the prose to what actually runs.
- **Acceptance check:** with a compatible directory, the branch prints Section 9's metric set and a reload-equivalence line with its tolerance, or the prose lists exactly the stages that run.
- **Spec:** DAT13, DAT14, VER4.

#### MXV-m2 — The BYOD image branch fails on a non-image with a raw exception, and the upload path can pick a stale file

- **Cell/section:** cell 29 image branch (`Image.open(image_path)`; `next(iter(sorted(_upload_into(...).iterdir())))`).
- **Observed issue:** a non-image raises `PIL.UnidentifiedImageError: cannot identify image file 'byod_inputs\\notes.png'` with no corrective action. The upload path takes the first sorted file in a persistent `outputs/byod/image/`, so a later upload whose name sorts later is ignored.
- **Consequence:** the learner gets an unhelpful error for the commonest mistake, and an earlier image can be reused silently.
- **Evidence:** direct (P4) for the exception; source for the upload ordering (dialog not run).
- **Recommended correction:** open through the dataset branch's decoding guard (or catch `OSError`) and name the accepted formats. Classify the files returned by this upload, not the directory listing.
- **Acceptance check:** a text file given to the image branch yields a `ValueError` naming the file and the accepted formats, and a second, different upload is the one classified.
- **Spec:** DAT19, UX10.

#### MXV-m3 — "Try next" asks for a runtime comparison nothing prints

- **Cell/section:** Interpretation "Try next" (template line 458).
- **Observed issue:** it asks the learner to "compare the held-out scores and runtime", but no cell prints a duration, and `finetune` returns none.
- **Consequence:** the comparison cannot be made without instrumenting the code.
- **Evidence:** source; P0 `finetune_returns_elapsed_time: false`.
- **Recommended correction:** print wall time in cell 19 and record it in the result JSON.
- **Acceptance check:** cell 19 prints the fine-tune duration.
- **Spec:** GDL10, UX5.

### Suggestions

- **MXV-S1 — Use the `ocarina` frog.** One sample frog's ImageNet top-1 is `ocarina` (0.302), with `tailed frog` third, yet the grouped-mass rule still classifies it correctly. Pointing at it would turn the zero-shot rule into an interpretation activity.
- **MXV-S2 — Cite the adapted head's blank/noise answers in the conclusion.** In the hosted run, `frog` 0.828 for a blank image and `truck` 0.756 for noise illustrate the no-reject-option limit.
- **MXV-S3 — Declare the current spec.** Regenerate against NOTEBOOK_SPEC 2.2 when the template is revised.

## 6. Readiness

**Needs revision.**

- **Open Majors:** MXV-M1 to MXV-M5.
- **Unresolved MUSTs:** RUN1, RUN10, ENV6 and REL2 (MXV-M1); SPL5 (MXV-M2). REL12 BYOD evidence is not recorded on a hosted runtime.
- **Remaining gates after the fixes:**
  - a one-pass hosted Run all of the regenerated blob;
  - a hosted BYOD positive and negative run (REL12);
  - a re-recorded comparison on a duplicate-aware split, read correctly whether or not zero-shot is at ceiling.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped):**
  - the full-scale split, duplicate overlap and zero-shot scores;
  - the default path end to end at reduced scale (`PER_CLASS = 30`);
  - the freeze-experiment failure (reduced scale);
  - the BYOD positive directory and five refusals.
- **Verified from documented execution:** the install-cell restart and the full-scale fine-tuned scores (Kaggle T4).
- **Inferred from source:** the upload-dialog behaviour (MXV-m2), the full-scale fine-tuned score on the copy-free subset, GPU behaviour of the experiment, and the effect of the proposed fixes.
- **Most likely to be wrong:** MXV-M4's severity. The notebook never states a numeric gain, and its "within noise" caveat is correct, so the defect could be read as a Minor framing problem. It is graded Major because the stated purpose of the baselines, the Section 7 claim and the only quantitative experiment all presuppose a gap the run does not show.

Probe ZIP: `maxvit_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
