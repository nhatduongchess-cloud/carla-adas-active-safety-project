# UFLDv2 security and compatibility audit

Date: 2026-09-03  
Status: **security scan passed; restricted runtime integration validated**

## Source and license

- Upstream: <https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2>
- Audited branch/ref: `master`
- Audited remote head: `c903880678454dfd9b55a63022368db05c00bc6d`
- License: MIT, copyright 2022 Zequn Qin.
- Repository scope: PyTorch lane-detection training, evaluation, demo and optional
  ONNX/TensorRT deployment utilities.

## Findings

Remote HEAD was re-verified with read-only `git ls-remote` on 2026-09-03 and
still matches the pinned commit above.

1. **No runtime auto-download is required by the core demo/train scripts.** The
   README links to Google Drive/Baidu checkpoints and a third-party model-zoo
   `download.sh`; those paths will not be executed automatically.
2. **Upstream checkpoints are unsafe to load as written.** `train.py`, `test.py`,
   `demo.py`, and `deploy/pt2onnx.py` call `torch.load(...)` without an explicit
   `weights_only=True`. A malicious pickle checkpoint could execute code. The
   project must never run those load sites on an untrusted `.pth` file.
3. **Dependencies are unpinned.** `requirements.txt` lists `opencv-python`,
   `tqdm`, `tensorboard`, `addict`, `sklearn`, `pathspec`, `imagesize`, and
   `ujson` without versions or hashes. The `sklearn` name is a deprecated meta
   package; use pinned `scikit-learn` only if the imported code truly needs it.
4. **The official install path includes native/privileged-risk steps that are not
   needed for this demo.** `INSTALL.md` proposes NVIDIA DALI from an extra index,
   `python setup.py install` for a custom interpolation extension, and optional
   C++/CMake evaluators. These broaden the supply-chain and native-code attack
   surface and are incompatible with a minimal Windows runtime.
5. **The config loader executes Python config files.** Only repository-pinned,
   reviewed configs may be used; user-supplied or downloaded config files must
   not cross this trust boundary.
6. Windows Defender custom scans of both the pinned source and checkpoint used
   signature `1.459.17.0` and reported zero new threats. After final runtime
   integration, the checkpoint and local backend were scanned again with newer
   signature `1.459.28.0` (updated 2026-09-03): threats before `0`, threats after
   `0`. The source contained 65 tracked files; static review found the unsafe
   load, shell and native-build sites listed above, none of which were executed.
7. `pip-audit` is not installed in the CARLA virtual environment, so no claim is
   made that a dependency CVE audit has passed. Installing an audit tool itself
   also requires approval under the project policy.

## Existing environment compatibility

The CARLA environment already contains PyTorch `2.7.1+cu118`, torchvision
`0.22.1+cu118`, OpenCV Python `5.0.0.93`, tqdm `4.70.0`, and scikit-learn
`1.9.0`. It does not contain tensorboard, addict, pathspec, imagesize, or ujson.
No missing package has been installed as part of this audit.

## Restricted integration

1. The pinned commit was downloaded into quarantine; no install hook ran.
2. The official Tusimple ResNet18 checkpoint was downloaded from the project's
   published Google Drive link. Its size was 385,540,040 bytes and SHA-256 was
   `490d8f81995cf6382005e037ca70b5260de6cff7da863c0965b2daeb3f710a4c`.
3. Do **not** install DALI, TensorRT, the native interpolation extension, the
   CULane evaluator, or the third-party model-zoo downloader for v1.
4. Reuse the current PyTorch/torchvision/OpenCV runtime. Add a missing Python
   package only if static import tracing proves it necessary, with a pinned
   version and a separate approval.
5. The checkpoint was opened only through `torch.load(..., map_location="cpu",
   weights_only=True)`. It contained one `model` dictionary with 126 tensors and
   strict-matched the reviewed Tusimple ResNet18 inference graph.
6. Keep learned-lane behind the existing confidence/geometry arbiter; map remains
   fallback and junction priority. Learned output cannot disable AEB/MRM.

## Decision

Remote and local review found no direct evidence of malicious behavior, but the
unpinned dependencies, native build steps, executable Python configs, and unsafe
upstream checkpoint loading make an unmodified `pip install -r requirements.txt`
or direct upstream `.pth` execution unacceptable. No dependency was installed.

The first CARLA attempt ended in a camera timeout and the restricted artifacts
were removed immediately, as required by the compatibility rule. A subsequent
model-free baseline reproduced the same CARLA server failure, proving that the
checkpoint was not the root cause. The server was using the much heavier
`Town10HD_Opt`; the validated low-memory path uses `Town02` and avoids redundant
same-map reloads. The user then explicitly requested that the audited model be
restored. Its SHA-256 matched the value above and a fresh Defender scan again
reported zero threats.

The active local backend imports no upstream package and executes none of its
install/config scripts. It retains the checkpoint's two ego-lane row heads,
which are exactly sliced from the strict-loaded output layer, and drops unused
outer-lane logits. YOLO and UFLDv2 run concurrently inside the same bounded
latest-frame job; learned-lane work is suspended while AEB is actively braking.
LiDAR/radar safety and map geometry therefore remain independent of this model.

Final real-CARLA low-memory acceptance on `Town02` completed two 600-frame runs:

- clear road: 100.31 m driven, 35.72 km/h maximum, zero collisions, safety p99
  17.40 ms and perception p95 47.52 ms;
- stationary hazard: zero movement/collisions, safety p99 17.02 ms and perception
  p95 39.90 ms.

Both runs had zero camera/LiDAR frame errors, 100% radar availability and zero
neural errors. The checkpoint remains in `weights/` but is excluded from Git;
runtime never downloads it. Learned control requires confidence ≥0.75 for five
unique valid frames, width 2.8–4.2 m and stable geometry. Three invalid frames,
stale output or any junction immediately returns control to CARLA map geometry;
Hough remains active in shadow/debug mode.
