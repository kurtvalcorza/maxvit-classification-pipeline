# Weight and sample-data provenance and hosting

- Upstream: `timm/maxvit_tiny_tf_224.in1k`
- Revision: **not yet pinned** (`MODEL_REVISION = "unpinned"`). Run `python tools/pin_snapshot.py` to resolve the Hub's `main` to a 40-hex commit, download every manifest-listed file at that commit, cross-check each LFS file against the SHA-256 the Hub records, record the Hub's LFS SHA-256 of the reference file, and write the commit and digests into the manifest and `src/maxvit_classification_pipeline/pipeline.py`.
- Executed artifact: `model.safetensors` (123,917,994 bytes as the Hub reported for `main` when this repository was built).
- Hosted, not executed: `pytorch_model.bin` (124,072,481 bytes), a pickle checkpoint with the same weights. It is listed under `referenceFiles`; the pin tool records its Hub LFS SHA-256 without downloading it.
- Manifest: `weights/maxvit-tiny-tf-224-in1k/dimer-base-manifest.json` (3 staged files: `README.md`, `config.json`, `model.safetensors`; `totalBytes` 123940709; plus the reference file). Every `sha256` is `null` until the pin tool runs.
- Committed copy: `config.json` (597 bytes) is committed as the Hub served it, so the offline tests can check the architecture, the tag, the fixed input size and the preprocessing against timm's built-in configuration. The pin tool replaces it with the bytes downloaded at the pinned commit before hashing.
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter; the upstream `google-research/maxvit` repository is also Apache-2.0). The model was trained on ImageNet-1k, whose images carry their own terms of access.
- Hosting: Apache-2.0 permits use, modification, distribution and commercial use, subject to keeping the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored).
- Fresh clone, once pinned: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: `timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=False, num_classes=1000)` builds the architecture without contacting the Hub, and `load_state_dict(..., strict=True)` loads the verified SafeTensors file. No pickle is deserialised and no model-repository code runs.

## Tutorial sample data

- Dataset: `Cleanlab/cifar-10-subset`, file `CIFAR-10-subset.zip`, at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`.
- Size and digest: 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`. `fetch_sample_archive` refuses any other bytes and has no fallback.
- Licence: MIT (the dataset card). The images are CIFAR-10 images (Krizhevsky, 2009); the tutorial keeps the `frog` and `truck` folders.
- Hosting: the archive is downloaded at runtime into a working directory (`data/`, git-ignored) and is not redistributed by this repository.
