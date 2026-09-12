# StopSign training-source security audit

Status: **POST-DOWNLOAD QUARANTINE GATE PASS — 2026-09-04**

This review covers the remaining Week-2 gap: visible `StopSign` examples for
the training split. CARLA-native visible signs are retained for Town05
validation and Town10HD held-out testing; no external validation/test images
will be mixed into those gates.

## Candidate

- Source: [ayoubsa/Sign_Road_Detection_Dataset](https://huggingface.co/datasets/ayoubsa/Sign_Road_Detection_Dataset)
- Pinned revision: `7073297e955abcc414ee24dde0f31648ce665ec7`
- Intended artifact: `train.zip` only (71.8 MB)
- Expected SHA-256: `5654556eddf2788df5603041bc4c866a350916e4bdc1128506689182c4e60a3d`
- Declared license: CC BY 4.0
- Declared task/format: object detection, YOLO annotations, 15 road-sign classes
- Required mapping: source class `Stop` -> ADAS taxonomy class 7 `StopSign`; all
  other source classes are excluded from this supplement.

## Static remote review

Hugging Face lists only `.gitattributes`, `README.md`, `train.zip`, `valid.zip`
and `test.zip` at the pinned repository revision. The host marks `train.zip`
as `Safe` and reports no problematic pickle imports. There are no advertised
install scripts, Python packages, model checkpoints or executable files, and
this project does not need to install a dependency to consume the data.

The dataset card says it was provided by a Roboflow user but does not link the
original Roboflow project or provide per-image provenance. This is a **medium
license/provenance risk**, not evidence of malware. For a portfolio/research
demo it can be used only with explicit attribution and this limitation
recorded; commercial redistribution requires a separate provenance review.

## Mandatory post-download quarantine gate

Approval to download is not approval to train. After download, the pipeline
must stop unless all checks below pass:

1. Match the archive SHA-256 exactly.
2. Run Microsoft Defender on the archive and quarantine directory.
3. Inspect the ZIP central directory before extraction: reject absolute paths,
   `..` traversal, symlinks, encrypted entries, duplicate normalized paths,
   suspicious compression ratios and any extension outside the explicit
   image/YOLO allowlist.
4. Extract without executing any file.
5. Reject malformed YOLO rows, invalid class IDs, non-finite coordinates,
   boxes outside `[0, 1]`, missing image/label pairs and corrupt images.
6. Hash images and reject duplicates against CARLA validation/test sources.
7. Visually inspect a seeded sample of StopSign boxes before promotion.
8. Record source revision, archive hash, scan result, class mapping and counts
   in the generated training manifest.

## Decision

The user explicitly approved the external training dataset. Only `train.zip`
was downloaded. Its SHA-256 matched the pinned value, the local archive audit
found 7,063 safe entries (3,530 JPG + 3,530 TXT) and no structural issue, and
Microsoft Defender reported **no threats**. No remote code was executed.

Source validation retained 282 unique StopSign training images after removing
three exact duplicates, remapped 282 `StopSign` and seven co-visible
`TrafficLight` boxes, decoded every selected image, and found no collision with
6,700 CARLA validation/test image hashes. The curated source is approved for a
small training supplement; the medium per-image provenance limitation remains
documented and must be disclosed in portfolio/commercial use.
