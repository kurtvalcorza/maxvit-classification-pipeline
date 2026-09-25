---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: image-classification
base_model: timm/maxvit_tiny_tf_224.in1k
date_published: "2022-04"
date_published_source: "month of the MaxViT paper (arXiv:2204.01697, submitted 2022-04-04); the date the timm port of the TensorFlow weights was first published is not established by this repository"
---

# MaxViT-Tiny (timm, ImageNet-1k, 224 px) — Image Classification with Bounded Fine-Tuning

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-timm%2Fmaxvit__tiny__tf__224.in1k-ffcc4d?style=flat)](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-google--research%2Fmaxvit-181717?style=flat&logo=github&logoColor=white)](https://github.com/google-research/maxvit)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2204.01697-b31b1b.svg)](https://arxiv.org/abs/2204.01697)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream snapshot is pinned to Hub commit `041f2cce4d74c7539d63aa9fb85786e78072d487`, and the manifest records every file's SHA-256. No execution with the pinned weights has been recorded yet, so this card claims no measured value for this repository.

---

## Interactive Colab Tutorials

- **End-to-end classification and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/maxvit-classification-pipeline/blob/main/tutorials/maxvit_classification_colab.ipynb) [`maxvit_classification_colab.ipynb`](https://github.com/kurtvalcorza/maxvit-classification-pipeline/blob/main/tutorials/maxvit_classification_colab.ipynb)
  *A pinned CIFAR-10 frog/truck subset, ImageNet top-5 predictions, majority-class and zero-shot ImageNet baselines, degenerate-input probes, a bounded fine-tune of a new two-class head, held-out and unseen-split scores, and SafeTensors adapter export and reload.*

---

#### Description

`timm/maxvit_tiny_tf_224.in1k` is MaxViT-Tiny from "MaxViT: Multi-Axis Vision Transformer" (Tu et al., arXiv:2204.01697). The upstream card states that it is an official MaxViT model, trained in TensorFlow on ImageNet-1k by the paper authors and ported to PyTorch by Ross Wightman; the `tf` in its name marks a model that matches the TensorFlow original exactly. The upstream card reports 30.9M parameters, 5.6 GMACs and a 224×224 input. A test in this repository counts 30,916,528 parameters in the architecture it builds.

MaxViT is a hybrid network. After a convolutional stem, four stages stack MaxViT blocks. Each block is an MBConv convolution (with BatchNorm and squeeze-and-excitation) followed by two self-attention steps: block attention inside non-overlapping 7×7 windows, for local interactions, and grid attention across a sparse 7×7 grid that spans the whole feature map, for global ones. The head applies LayerNorm to the pooled features, a 512-wide Tanh layer and a linear layer (`head.fc`) that outputs a softmax over the 1000 ImageNet-1k classes.

The window and grid partitions fix the input size at 224×224. The checkpoint's preprocessing resizes the image with bicubic interpolation so its shorter side is 235 px, centre-crops 224×224 (`crop_pct` 0.95), and normalises with the ImageNet mean and standard deviation.

This repository adds gradient fine-tuning on a caller's labelled images. `from_pretrained(class_names=...)` replaces the 512-to-1000 `head.fc` layer with a new layer for the caller's classes. `finetune` then trains with cross-entropy, either the whole network (the default) or the head alone (its LayerNorm, Tanh layer and new `fc`) with the backbone frozen.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the pinned files, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `MaxViTPipeline.from_pretrained`: construction with `timm.create_model(..., pretrained=False)`, so timm downloads nothing itself, then `load_state_dict(strict=True)` from the verified SafeTensors file;
- `predict`: input checks and top-k softmax scores; `zero_shot_evaluate`: a baseline that maps groups of ImageNet classes onto task labels without training;
- `fetch_sample_archive`, `read_class_archive`, `read_class_folder`, `validate_dataset`, `split_dataset`, `validate_inputs` and `evaluation_report`: the data acquisition, validation and single-batch evaluation stages;
- `finetune`, `evaluate` (accuracy, balanced accuracy, per-class recall, confusion matrix, majority-class baseline), `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is closed-set, single-label image classification. `predict` takes one image or a batch of up to 64 and returns, per image, the top-k classes with their softmax scores and the argmax label.

The pretrained head fits photographs of the objects, animals and scenes ImageNet-1k names. The adaptation path fits a small labelled set of a few classes, for example product categories, plant or specimen types, or document page types, where a new head on a pretrained backbone is enough.

The intended role is a reference hybrid convolution-attention classifier and a teaching baseline for transfer learning. MaxViT-Tiny is small enough to fine-tune on a free GPU in minutes. A reader's own application can embed `MaxViTPipeline` as a classifier or, with the head removed, as an image feature extractor.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- the pretrained vocabulary is the 1000 ImageNet-1k classes; anything else receives the nearest of them;
- a `score` is a softmax value under the model's training distribution, not a calibrated probability on the user's images;
- the classifier has no reject option: every image, including a blank one, receives a class;
- accuracy is only meaningful next to the majority-class baseline of the same data, and balanced accuracy is the fairer summary when classes are imbalanced;
- sketches, screenshots, medical, aerial and thermal images, and very small images upsampled to 224 px, are distribution shifts from ImageNet photographs;
- a fine-tune on a few hundred images demonstrates the workflow and does not produce a deployable classifier.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside the loaded vocabulary, no text prompts, no multi-label output, no localisation (detection or segmentation), and no calibrated confidence or open-set rejection.
2. **Input boundary:** `predict` and `validate_inputs` reject anything that is not a `PIL.Image.Image` (`TypeError`), batches above `MAX_BATCH = 64`, and image sides below `MIN_IMAGE_SIDE = 8` px or above `MAX_IMAGE_SIDE = 4096` px (`ValueError`). `top_k` must be between 1 and the number of classes.
3. **Input boundary:** every image is resized so its shorter side is 235 px and centre-cropped to 224×224; the architecture accepts no other input size. Content outside the central crop is not seen, and small images are upsampled. The tutorial's 32×32 CIFAR-10 images are an example of the latter, not a recommended input size.
4. **Data boundary for adaptation:** `validate_dataset` accepts up to 5,000 records (`MAX_RECORDS`) over 2 to 1,000 classes, with at least 2 images per class. The bounded tutorial fine-tune (5 epochs, no augmentation) is not a training recipe for a production classifier.
5. **Decision boundary:** not for decisions that act on labels without a person reviewing them, including medical, safety, legal, hiring, credit or law-enforcement decisions. Any use also requires accuracy and per-class recall measured locally on the deployment's own labelled images.

#### Factors

###### Groups

ImageNet-1k names only three person classes (`ballplayer`, `groom` and `scuba diver`), but many of its images show people, and classes such as clothing, sports equipment and musical instruments correlate with them. The pretrained model can therefore produce labels whose errors differ with the skin tone, age, gender presentation or dress of the people in an image; this was not evaluated here or, as far as this repository knows, upstream.

ImageNet's photographs were collected from web search in a limited set of languages and regions. Objects, foods, animals and scenes from under-represented regions may be recognised less often; that is unmeasured.

An adapted model inherits whatever group structure the caller's labelled data has. The operator who classifies images of people, with the pretrained or an adapted model, owns a per-group audit on their own images before relying on the output.

###### Instrumentation

ImageNet-1k images are web photographs of varied resolution, mostly taken with consumer cameras. Inference images arrive from whatever produced them: phones, scanners, microscopes, drones or rendering engines.

Resolution, blur, compression, exposure, colour balance and framing all change the evidence. The resize-and-crop preprocessing discards the border of every non-square image and resamples every image to 224 px. The pipeline checks only type and size; it cannot detect a rotated scan, a thumbnail, a rendered image or an empty frame.

The tutorial sample is itself an instrument: CIFAR-10 images are 32×32 pixels, collected by Krizhevsky and labelled by hand, and are upsampled about sevenfold here. Labels a caller supplies carry whatever error their annotation process has, and a fine-tune learns a systematic labelling error as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `timm==1.0.29`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. No run with the pinned weights has been recorded yet, so no runtime, memory or throughput figure is given.

**Data environment.** The pretrained head assumes a photograph centred on one ImageNet object or scene. An adapted model assumes inference images that resemble its training images in source, framing and resolution. The tutorial's adaptation data is CIFAR-10 thumbnails, so a model adapted on it transfers to images of that kind and to little else. When these assumptions fail, the model still returns a class. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`evaluate(records, majority=...)` reports, for this pipeline's head on labelled records:

- `accuracy`: the fraction of images whose argmax label is correct;
- `balanced_accuracy`: the mean of the per-class recalls, which a model cannot raise by favouring the larger class;
- `per_class_recall` and `confusion_matrix` (rows true, columns predicted), which show where the errors go;
- `majority_baseline_accuracy`: the accuracy of always answering `majority`, which the caller sets to the training split's most frequent class.

`zero_shot_evaluate(records, groups)` scores the unmodified ImageNet head on task labels. It assigns each image to the label whose group of ImageNet classes holds the most softmax mass, and reports the same measures. For the tutorial, `frog` maps to ImageNet's three frog classes and `truck` to its big-truck classes; pickups are left out because CIFAR-10's `truck` excludes them.

`evaluation_report(result, truth, groups=...)` covers one batch of ImageNet-head predictions. It reports how often the true group appears in the top-1 and the top-k, with the verdict `sample-sanity`. Without labels it returns `not-measurable` and names the labelled data that would be needed.

The upstream card's comparison table reports ImageNet top-1 accuracy 83.41 and top-5 accuracy 96.59 for this model. Those values are upstream-reported, and this repository does not reproduce them. No value from this repository has been recorded yet.

###### Decision thresholds

There is no score threshold. The reported label is the argmax over the head's classes, so every image receives exactly one of them. `top_k` changes how many ranked classes are returned, not which one is reported.

No acceptance threshold on accuracy is set anywhere in the repository. A deployment that needs to abstain must add its own rule, for example a minimum top-1 score chosen on labelled images from the deployment, and must measure what that rule costs in coverage. Re-check any such rule after a change of camera, image source, class set or adapter.

###### Approaches to uncertainty and variability

Every score is one pass over one split: no repeated runs, no cross-validation, no bootstrap and no confidence interval. With the default 200 images per class, the tutorial's held-out split has 60 images, so one image moves accuracy by about 1.7 percentage points, and differences of a few points between methods are within noise.

Sources of run-to-run variability:

- the per-class sample drawn from the archive and the split, controlled by `DATASET_SEED` and `SEED`;
- the head initialisation and the batch order, controlled by the `seed` argument;
- BatchNorm statistics in the MBConv blocks, which update during full fine-tuning (timm builds this model with `drop_rate` 0.0 and no stochastic depth, so there is no dropout);
- GPU kernel selection, which is not forced to be deterministic, so repeated GPU runs can differ slightly.

A `score` is a softmax output, not a calibrated probability. A caller who needs calibrated confidence must fit a calibration map on labelled images from the deployment. A caller who needs an uncertainty estimate must evaluate on more images, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The upstream card states that the model was trained on ImageNet-1k, in TensorFlow, by the paper authors. ImageNet's images were gathered by web search and labelled by crowd workers; they include identifiable people and private settings, and published audits of the wider ImageNet collection have documented offensive and stereotyped labels in its person categories. Personal data is present in the training corpus by construction; it was not audited here.

The tutorial downloads `CIFAR-10-subset.zip` from the `Cleanlab/cifar-10-subset` dataset at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75` (MIT licence, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`). It keeps the `frog` and `truck` folders. CIFAR-10 is a set of 32×32 colour images labelled with ten object classes.

This repository distributes code, tests, documentation and one configuration file copied from the upstream snapshot (`config.json`). It does not distribute `model.safetensors` or the sample archive; both are downloaded at runtime and git-ignored.

An exported adapter contains weights fitted to the caller's training images. It does not contain the images, but it can reflect them. The operator must audit the images they classify, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, the upstream authors, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: triage of medical or dermatology photographs, sorting of images of people, content moderation, and quality inspection that stops a production line. Any of them would be admissible only with human review of every acted-on label. They would also need locally measured accuracy and per-class recall stratified by the groups named above, a documented abstention and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** while `MODEL_REVISION` is `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` raise before any download or model import. Once pinned, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants. It fetches only manifest-listed files, and only with `allow_download=True`. `verify_snapshot` checks every file's byte size and SHA-256 and refuses an entry with no recorded digest. `from_pretrained` builds the architecture with `pretrained=False` and loads the verified SafeTensors file with `strict=True`. The upstream `pytorch_model.bin` pickle is never staged or loaded; the pin tool records its Hub LFS digest for provenance only.
- **Data integrity:** `fetch_sample_archive` downloads the sample at a fixed dataset commit and checks its size and SHA-256 before it is opened, with no fallback. `read_class_archive` refuses absolute member names and `..` segments and bounds the member count and the uncompressed size before decompressing anything; `read_class_folder` refuses files that link outside the directory.
- **Tests of those refusals:** tests assert that an unpinned package, a missing snapshot and a tampered digest are all refused before `torch`, `timm` or `safetensors` is imported. Others assert that the committed `config.json` agrees with the manifest size and with timm's built-in preprocessing, that the ImageNet groups name the classes stated above, and that a frozen fine-tune leaves every backbone weight and BatchNorm statistic unchanged.
- **Input integrity:** `validate_inputs` and `predict` share one checker for type, batch size, image size and `top_k`. `validate_dataset` rejects a record with missing keys, a non-image, an out-of-range image, an unknown label or a class with fewer than 2 images. It reports class imbalance and pixel-identical duplicates as findings.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-weights digest, class names and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key. It also refuses a different vocabulary, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for sampling, split, head initialisation and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no pickle deserialisation, no remote model code, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no class balancing, re-sampling or augmentation; `validate_dataset` reports imbalance and does not change it.

###### Risks and harms

- **A class for every image.** Blank, corrupted and out-of-domain images receive a label, often with a high score. Whoever acts on that label bears the harm; the tutorial probes a blank and a noise image and records what it finds.
- **Mislabelling within a closed vocabulary.** An object outside the vocabulary that resembles a class is labelled as that class, and nothing flags it.
- **Unequal error across groups.** Errors on images of people, and on objects from under-represented regions, may be more frequent; the people in the images bear that harm. It is unmeasured.
- **Overfitting in adaptation.** A fine-tune on a few hundred images can score well on a held-out split from the same source and fail on anything else. The operator who deploys it bears the harm whenever training and deployment images differ.
- **Misleading accuracy.** Accuracy on an imbalanced set can exceed the majority baseline by little while looking high. Reporting it without the baseline and balanced accuracy overstates quality.
- **Automation bias.** High softmax scores invite trust that an uncalibrated score has not earned. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of records that share a source photograph or session puts near-duplicates on both sides. The resulting held-out score overstates quality; `validate_dataset` reports exact duplicates, and `split_dataset` documents that grouped data must be split by group.

###### Use cases

The following uses are prohibited even where the model would work:

- classifying people in order to surveil, track, profile or score them, or to infer sensitive attributes;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present labels as verified facts or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed labels;
- any use that violates the upstream Apache-2.0 licence, the MIT licence of the sample archive, or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `timm/maxvit_tiny_tf_224.in1k`
- Revision: `041f2cce4d74c7539d63aa9fb85786e78072d487` (pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest file at it, recorded each file's SHA-256, and recorded the Hub's LFS SHA-256 of the reference file without downloading it).
- Snapshot manifest: `weights/maxvit-tiny-tf-224-in1k/dimer-base-manifest.json`, 3 staged files, `totalBytes` 123940709, plus one reference file. The byte sizes and digests describe the files at the pinned commit.
- `model.safetensors` (executed artifact): 123,917,994 bytes; SHA-256 `e3998ec8e5f70ad5fa7682a1fa211b76d4adcb775f809b976b1a60506be113e6` (matches the Hub's LFS record).
- `pytorch_model.bin` (pickle of the same weights, reference only): 124,072,481 bytes; never staged or loaded; Hub LFS SHA-256 `bb7df98fcb8576411cd97b34fdbb418bed22710fbf60943ba943fe46d332ff8b`.
- `config.json`: 597 bytes; `maxvit_tiny_tf_224`, tag `in1k`, 1000 classes, fixed 224×224 input, bicubic, `crop_pct` 0.95, ImageNet mean and standard deviation, classifier `head.fc`.
- `README.md`: 22,118 bytes; the upstream model card.
- Loader: `timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=False, num_classes=1000)`, then `load_state_dict(safetensors.torch.load_file(<verified file>), strict=True)`.
- Sample dataset: `Cleanlab/cifar-10-subset` at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`, `CIFAR-10-subset.zip`, 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`.

## Input/output contract

- `MaxViTPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=20260925)`: stage (only with `allow_download=True`), verify, load; with `class_names`, replace the head deterministically under `seed`.
- `predict(images, top_k=None) -> dict`: keys `predictions` (per image: `predicted_label`, `predicted_index`, `top_k` as a list of `{"label", "index", "score"}` in descending score), `top_k`, `decision_rule` (`"argmax"`), `class_names_count`, `adapted`, `device`, `model_id`, `model_revision`. `top_k` defaults to 5, or to the number of classes if smaller.
- `zero_shot_evaluate(records, groups=IMAGENET_GROUPS, *, majority=None) -> dict`: the measures of `evaluate`, plus `rule`, `groups` and `mean_group_mass`; ImageNet head only.
- `finetune(records, *, epochs=5, batch_size=16, learning_rate=1e-4, weight_decay=0.01, seed=20260925, freeze_backbone=False, progress=None) -> dict`: AdamW, cross-entropy, float32, no augmentation; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, majority=None) -> dict`: `accuracy`, `balanced_accuracy`, `per_class_recall`, `support`, `confusion_matrix`, `n`, `mean_top1_score`, `majority_baseline_accuracy` (with `majority`), `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `maxvit-adapter-v1`.
- Records: `{"id": str, "image": PIL.Image.Image, "label": str}`; `read_class_archive(zip)` and `read_class_folder(directory)` read them from `<class>/<image>` layouts.
- Constants: `MIN_IMAGE_SIDE = 8`, `MAX_IMAGE_SIDE = 4096`, `MAX_BATCH = 64`, `NUM_IMAGENET_CLASSES = 1000`, `DEFAULT_TOP_K = 5`, `IMAGENET_GROUPS` (frog: 30, 31, 32; truck: 555, 569, 675, 864, 867), `MAX_RECORDS = 5000`, `MIN_PER_CLASS = 2`.

## Verification records

No execution with the pinned weights has been recorded. The offline test suite runs the `maxvit_tiny_tf_224` architecture, shrunk to one 32-wide block per stage, with random weights through prediction, the zero-shot baseline, full and frozen fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Tu, Talebi, Zhang, Yang, Milanfar, Bovik and Li. MaxViT: Multi-Axis Vision Transformer. ECCV 2022. https://arxiv.org/abs/2204.01697
- Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.
- Upstream code and weights: https://github.com/google-research/maxvit (Apache-2.0)
- PyTorch port: https://github.com/huggingface/pytorch-image-models
- Upstream card: https://huggingface.co/timm/maxvit_tiny_tf_224.in1k
- Sample dataset: https://huggingface.co/datasets/Cleanlab/cifar-10-subset
