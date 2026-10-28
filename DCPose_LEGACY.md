# DCPose Video Inference Guide

This guide explains how to run the repository's official DCPose video pipeline on a video clip. The pipeline is implemented in [`demo/video.py`](./demo/video.py) and performs:

1. Video-to-frame extraction.
2. YOLOv3 person detection on every frame.
3. DCPose inference using the current, previous, and next frames for each detected person.
4. PoseTrack keypoint-order conversion.
5. JSON, pose-overlay image, bounding-box image, and annotated-video output.

## Step-by-step instructions

### 1. Enter the DCPose project

```bash
cd /home/lunet/cogt/Project/Pose-Estimation/DCPose
```

Run the demo from the directories expected by its relative paths. The official video command is run from `DCPose/demo`.

### 2. Create the recommended environment

The code was developed with Python 3.6.12, PyTorch 1.4.0, torchvision 0.5.0, CUDA 10.0, and Ubuntu 18.04. The pinned dependencies are listed in [`DCPose_requirement.txt`](./DCPose_requirement.txt).

```bash
conda create -n DCPose python=3.6.12
conda activate DCPose
pip install -r DCPose_requirement.txt
```

These are old dependencies. On a modern system, installing them may require an older CUDA-compatible environment or container.

### 3. Build the custom deformable-convolution extension

DCPose uses the CUDA extensions in [`thirdparty/deform_conv/`](./thirdparty/deform_conv/).

```bash
conda activate dcpose_legacy_env

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="7.5"

cd thirdparty/deform_conv
rm -rf build deform_conv.egg-info
python setup.py develop
cd ../..
```

The build requires a working CUDA toolkit, `nvcc`, and a PyTorch installation compatible with that toolkit. The legacy environment uses GCC/G++ 8.4 because CUDA 10.1 rejects newer host compilers. The repository's [`thirdparty/deform_conv/setup.py`](./thirdparty/deform_conv/setup.py) passes `CUDAHOSTCXX`/`CXX` to NVCC with `-ccbin`; this is necessary because the old PyTorch 1.4 build machinery may otherwise invoke a bare system `gcc`.

Optional import check:

```bash
python -c "import torch; import deform_conv_cuda; import deform_pool_cuda; print('deformable convolutions loaded')"
```

Import PyTorch before either extension. These legacy extensions depend on symbols exported by the loaded PyTorch runtime, so importing an extension first can produce undefined-symbol errors such as `THPVariableClass` or `c10::BFloat16`.

### 4. Download the pretrained supplementary files

Download and extract the supplementary files linked from [`README.md`](./README.md). For video inference, the important files are expected at:

```text
DcPose_supp_files/
├── object_detector/
│   └── YOLOv3/
│       └── yolov3.weights
└── pretrained_models/
    └── DCPose/
        └── PoseTrack17_DCPose.pth
```

The detector path is used by [`object_detector/YOLOv3/detector_yolov3.py`](./object_detector/YOLOv3/detector_yolov3.py). The pose checkpoint is the default used by [`tools/inference.py`](./tools/inference.py).

The detector and pose checkpoint are separate artifacts. The bundled detector constructs the repository's Darknet-style YOLOv3 model from [`object_detector/YOLOv3/config/yolov3.cfg`](./object_detector/YOLOv3/config/yolov3.cfg) and calls `load_darknet_weights()` for a `.weights` file. An Ultralytics YOLOv3 download that supplies a `.pt` checkpoint is not a drop-in replacement for this legacy loader; using it requires replacing or adapting the detector implementation. A matching original Darknet-format `yolov3.weights` file can be used with the bundled detector, provided its configuration and class layout match the bundled `yolov3.cfg`.

The file that cannot be substituted is `PoseTrack17_DCPose.pth`. It is the DCPose pose-estimation checkpoint and must come from the DCPose authors or a legitimate archived/rehosted copy of the exact model. Generic YOLO, Ultralytics, HRNet, or other pose checkpoints cannot replace it without changing the model and inference pipeline.

**Checkpoint recovery note:** On 2026-10-28, the project owner reached out to Liu Zhenguang, a DCPose author, for advice or an alternate source for the pretrained DCPose checkpoint weights. Until a response or replacement archive is available, legacy reproduction remains blocked and modernization is the active path.

### 5. Link the supplementary directory into the project

If the extracted supplementary directory is `/home/lunet/models/DcPose_supp_files`, create the expected link:

```bash
cd /home/lunet/cogt/Project/Pose-Estimation/DCPose
ln -s /home/lunet/models/DcPose_supp_files DcPose_supp_files
```

Verify the two files needed by the video demo:

```bash
test -f DcPose_supp_files/object_detector/YOLOv3/yolov3.weights
test -f DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth
```

PoseTrack image datasets and annotation files are not required for this video demo; they are used for training and evaluation.

### 6. Select a usable GPU

The detector currently contains:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
```

in [`object_detector/YOLOv3/detector_yolov3.py`](./object_detector/YOLOv3/detector_yolov3.py). This forces the detector to use physical GPU 3. Remove or comment out that line, then select the GPU when launching:

```bash
CUDA_VISIBLE_DEVICES=0 python video.py
```

Alternatively, change the hardcoded value to the desired physical GPU. The pose inference code calls `.cuda()` directly, so the current official demo requires a CUDA-capable GPU.

### 7. Put the video in the input directory

```bash
cd /home/lunet/cogt/Project/Pose-Estimation/DCPose
mkdir -p demo/input
cp /absolute/path/to/your_clip.mp4 demo/input/
```

The demo scans `demo/input` for lowercase `.mp4` and `.mp3` filenames. Use a lowercase `.mp4`; `.MP4`, `.mov`, and `.avi` are not picked up without changing [`demo/video.py`](./demo/video.py). The `.mp3` entry appears to be a typo and does not provide useful video support.

### 8. Run the official DCPose video demo

```bash
cd /home/lunet/cogt/Project/Pose-Estimation/DCPose/demo
CUDA_VISIBLE_DEVICES=0 python video.py
```

The default pose configuration is [`configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml`](./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml), and the default checkpoint is the PoseTrack17 DCPose checkpoint.

### 9. Inspect the generated outputs

For `demo/input/my_clip.mp4`, the pipeline creates numbered JPEG frames and outputs similar to:

```text
demo/input/my_clip/
├── 00000000.jpg
├── 00000001.jpg
└── ...

demo/output/
├── json/
│   └── result_my_clip.json
├── vis_img/
│   └── my_clip/
├── vis_img_box/
│   └── my_clip/
└── my_clip.mp4
```

The JSON contains per-detection image paths, `x/y/width/height` bounding boxes, and 17 predicted keypoints represented as `[x, y, confidence]`.

## Concerns and smoke tests

Run the smoke test in the indicated step before running a long clip. A passing test does not guarantee good pose quality, but it identifies whether the corresponding setup concern is likely to block inference.

| Concern | Instruction step | Smoke test | What indicates a problem | Solution / status |
|---|---:|---|---|---|
| Python/PyTorch/CUDA version incompatibility | 2 | `python --version && python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"` | Import errors, missing CUDA, or a PyTorch/CUDA combination incompatible with the installed driver/toolkit. | **PASSED in `dcpose_legacy_env`.** Verified Python 3.6.12, torch 1.4.0, torchvision 0.5.0, CUDA build 10.1, and CUDA available on the NVIDIA GeForce RTX 5090. The base Python 3.12.7 environment is intentionally not used for legacy validation. |
| Required Python packages are missing | 2 | `python -c "import cv2, numpy, PIL, yaml, torch, torchvision, tqdm; print('runtime imports OK')"` | Any `ModuleNotFoundError` or binary import error. | **PASSED in `dcpose_legacy_env`.** Verified cv2 4.4.0, NumPy 1.19.2, Pillow 7.2.0, PyYAML 5.3.1, tqdm 4.49.0, torch 1.4.0, and torchvision 0.5.0. The base Python 3.12.7 environment remains reserved for future modernization work. |
| CUDA toolkit/compiler is unavailable for the custom extension | 3 | `nvcc --version` | `nvcc` is missing or reports a toolkit incompatible with the installed PyTorch CUDA build. | **TOOLKIT AVAILABLE.** The Conda environment exposes `/home/lunet/cogt/.conda/envs/dcpose_legacy_env/bin/nvcc`, reporting CUDA 10.1, V10.1.243, matching the legacy PyTorch CUDA build. The NVIDIA driver version reported by `nvidia-smi` is not a substitute for an installed toolkit. CUDA 10.1 rejects host compilers newer than GCC 8, so the environment uses the matching Conda GCC/G++ 8.4 toolchain (`x86_64-conda-linux-gnu-gcc` and `x86_64-conda-linux-gnu-g++`). `CC`, `CXX`, and `CUDAHOSTCXX` point directly to those prefixed executables; `which gcc` may still show `/usr/bin/gcc` because it only searches `PATH`. The `cc1plus` backend exists and passes a standalone C++ smoke test. The extension setup passes the selected host compiler to NVCC with `-ccbin`, avoiding the old PyTorch build path's fallback to system `gcc`. `TORCH_CUDA_ARCH_LIST=7.5` is required as a compatibility build experiment because PyTorch 1.4/CUDA 10.1 cannot identify the RTX 5090's native architecture; this does not provide native RTX 5090 support. |
| Deformable-convolution extension did not build or cannot load | 3 | `python -c "import torch; import deform_conv_cuda; import deform_pool_cuda; print('deformable convolutions loaded')"` | `ModuleNotFoundError`, undefined symbols, or a shared-library loading error. | **COMPILED.** Both shared libraries were produced after selecting GCC/G++ 8.4 and passing `-ccbin` to NVCC. Import PyTorch before the extensions because the legacy modules depend on symbols exported by the loaded PyTorch 1.4 runtime; importing them first produced undefined `c10::BFloat16`/`TypeMeta` and `THPVariableClass` symbols. The correct smoke test is therefore `import torch` followed by `import deform_conv_cuda` and `import deform_pool_cuda`. Compilation/import success still requires a subsequent actual CUDA operator/model execution test on the RTX 5090; `sm_75` is only a compatibility experiment, not native architecture validation. |
| Supplementary model files are missing or placed incorrectly | 4–5 | `test -f DcPose_supp_files/object_detector/YOLOv3/yolov3.weights && test -f DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth && echo 'model files found'` | Either `test` command fails, or the symlink points to the wrong directory. | **PARTIALLY UNBLOCKED.** The detector weights can be sourced independently in the original Darknet `.weights` format and paired with the repository's bundled `yolov3.cfg`; an Ultralytics `.pt` model is not directly compatible with `detector_yolov3.py`. The DCPose-specific `PoseTrack17_DCPose.pth` checkpoint remains blocked because the official Google Drive folder linked by [`docs/Installation.md`](./docs/Installation.md) (`1VPcwo9jVhnJpf5GVwR2-za1PFE6bCOXE`) returns HTTP 404. A detector-only download is not a complete DCPose setup, and generic pose or YOLO checkpoints cannot replace the DCPose checkpoint without changing the model/pipeline. |
| Training assets are missing even if the DCPose checkpoint is unavailable | Training | `test -f DcPose_supp_files/pretrained_models/pretrained_coco_model.pth && test -f DcPose_supp_files/posetrack17_json_files/posetrack_train.json && test -d DataSet/PoseTrack2017/images_renamed` | The training configuration cannot initialize its HRNet backbone or construct the PoseTrack temporal training samples. | **TRAINING IS POSSIBLE IN PRINCIPLE, BUT NOT YET UNBLOCKED.** The training config uses `MODEL.PRETRAINED`=`pretrained_coco_model.pth`, and `DcPose_RSN.init_weights()` raises if that file is missing; the missing `PoseTrack17_DCPose.pth` is the final trained DCPose model, not the only required weight. Training also requires PoseTrack17 images, the repository-compatible COCO-format JSON files, and the supplementary annotation/box metadata. A true random-initialization experiment would require an intentional config/code change (`MODEL.PRETRAINED` empty or `MODEL.INIT_WEIGHTS` adjusted), plus lower batch size and a new validation plan; it would not reproduce the published pretrained result. |
| The supplementary directory symlink is broken | 5 | `readlink -f DcPose_supp_files && find -L DcPose_supp_files -maxdepth 4 -type f | sort | sed -n '1,30p'` | The resolved path does not exist or the expected weights are absent. | **WAITING FOR AN ALTERNATE ARCHIVE.** Do not create the symlink until the files have been obtained and their layout verified. Once an alternate archive is available, place it in a path such as `/home/lunet/models/DcPose_supp_files`, verify both required weights, then create the absolute symlink. |
| The detector is forced onto nonexistent GPU 3 | 6 | `nvidia-smi --query-gpu=index,name --format=csv` and inspect the hardcoded `CUDA_VISIBLE_DEVICES` line in `object_detector/YOLOv3/detector_yolov3.py` | There is no GPU 3, or the detector uses a different GPU than the one selected at launch. | **Pending smoke test.** Remove the detector's hardcoded GPU override and select the device with `CUDA_VISIBLE_DEVICES`. |
| The selected GPU is not visible or usable | 6 | `CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"` | Device count is zero or CUDA initialization fails. | **Pending smoke test.** Resolve the driver/CUDA/PyTorch mismatch or select an available GPU. |
| The official inference path is accidentally run on CPU | 6 | `python -c "import torch; print(torch.cuda.is_available())"` | `False`; the current code calls `.cuda()` and will not run as-is on CPU. | **Pending smoke test.** Use a CUDA GPU for legacy inference; modernisation should replace direct `.cuda()` calls with explicit device handling if CPU support is required. |
| The video is not discovered because of its extension or location | 7 | `cd demo && python -c "from utils.utils_folder import list_immediate_childfile_paths; print(list_immediate_childfile_paths('./input', ext=['mp3','mp4']))"` | The expected video is not listed. Rename it to lowercase `.mp4` or update the extension filter. | **Pending smoke test.** Put the clip in `demo/input` and use lowercase `.mp4`, or update the filter for the desired formats. |
| The video cannot be decoded by OpenCV | 7 | `python -c "import cv2; p='demo/input/your_clip.mp4'; c=cv2.VideoCapture(p); print('opened:', c.isOpened(), 'frames:', int(c.get(cv2.CAP_PROP_FRAME_COUNT)), 'fps:', c.get(cv2.CAP_PROP_FPS)); c.release()"` | `opened: False`, zero frames, or an invalid FPS. | **Pending smoke test.** Convert the clip to a supported codec/container or repair the OpenCV video installation. |
| The command is run from the wrong working directory | 8 | `cd demo && python -c "import os; print(os.getcwd()); print(os.path.isdir('./input'))"` | `input` is not found or relative checkpoint/config paths resolve incorrectly. | **Pending smoke test.** Run the official demo from `DCPose/demo` and use `PYTHONPATH=..` for isolated import checks. |
| The pose checkpoint/configuration pair is wrong | 8 | `cd demo && PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python -c "import tools.inference as i; print(type(i.model).__name__)"` | Checkpoint-not-found errors, missing/unexpected state-dict keys, or model construction errors. | **Pending smoke test.** Use the PoseTrack17 inference YAML with its matching checkpoint, then document any state-dict adaptation. |
| The repository has no PoseTrack18 inference YAML for the demo | 8 | `test -f configs/posetimation/DcPose/posetrack18/model_RSN_inference.yaml` | The test fails; use the supplied PoseTrack17 inference configuration/checkpoint unless you create a compatible PoseTrack18 inference setup. | **Known repository gap.** Use PoseTrack17 for the supplied demo, or add and validate a PoseTrack18 inference configuration/checkpoint pair. |
| The output directory cannot be created or written | 8–9 | `cd demo && mkdir -p output/vis_img output/vis_img_box output/json && touch output/.write_test && rm output/.write_test` | Permission denied or a filesystem/storage error. | **Pending smoke test.** Ensure the project/output location is writable and has sufficient free space. |
| The output codec is unavailable | 9 | `python -c "import cv2, tempfile, os; p=os.path.join(tempfile.gettempdir(), 'dcpose_codec_test.mp4'); w=cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'DIVX'), 25, (64,64)); print('opened:', w.isOpened()); w.release(); os.remove(p) if os.path.exists(p) else None"` | `opened: False`; change the codec in `utils/utils_video.py` to one supported by the local OpenCV build, such as `mp4v`. | **Pending smoke test.** Test `DIVX`; if unavailable, switch the writer to a codec supported by the local OpenCV build. |
| Output playback speed differs from the source | 9 | Compare `CAP_PROP_FPS` for the input with the output; the current `image2video` implementation always writes at 25 FPS. | Different source/output FPS or visibly incorrect playback speed. Pass the source FPS through to `image2video`. | **Known implementation limitation.** Preserve the source FPS when modernising the video writer; accept 25 FPS only for the legacy control. |
| Multiple people do not retain stable identities | 8–9 | Use a short multi-person clip and inspect `output/json/result_<name>.json` for detections across adjacent frames. | Detection order changes between frames; the official demo has no person tracker. Add tracking if persistent identities are required. | **Known implementation limitation.** Add a tracker only when stable identities are required, and evaluate it separately from pose quality. |
| The clip is too long for a first run or storage is insufficient | 7–9 | Run a very short clip first and check `df -h .`; estimate one JPEG per frame plus visualization/output storage. | Disk fills, processing is unexpectedly slow, or the run fails partway through. | **Pending smoke test.** Start with a three-to-ten-frame clip, confirm outputs, then scale up while monitoring disk and runtime. |
| Frame boundary handling behaves unexpectedly | 8 | Run a clip of at least three frames and inspect the first and last frame outputs. | The implementation reuses the boundary frame when there is no previous or next frame; this is expected, but very short clips provide little temporal context. | **Expected behavior.** Retain boundary-frame reuse for legacy compatibility; document any changed temporal padding in modernisation. |

## Legacy compiler and extension lessons

The following facts were established while bringing up the legacy build on the RTX 5090:

1. **Driver CUDA and toolkit CUDA are different things.** `nvidia-smi` reports the CUDA version supported by the installed NVIDIA driver; it does not prove that `nvcc` is installed. PyTorch can use its bundled CUDA runtime without `nvcc`, but compiling `deform_conv_cuda` requires the toolkit.
2. **The working legacy combination is deliberately old.** The validated environment uses Python 3.6.12, PyTorch 1.4.0, torchvision 0.5.0, and CUDA 10.1. This is a reproducibility environment, not the recommended basis for modernization.
3. **CUDA 10.1 requires an old host compiler.** Its compiler checks reject GCC versions newer than 8. GCC/G++ 8.4 must be installed as a matching pair; mixing GCC 8 frontends with newer runtime/compiler packages can cause confusing failures.
4. **Conda compiler names are prefixed.** The usable executables are `x86_64-conda-linux-gnu-gcc` and `x86_64-conda-linux-gnu-g++`. Therefore `which gcc` may still report `/usr/bin/gcc` even when `CC` and `CXX` correctly point to GCC 8. `which` searches `PATH`; it does not inspect those variables.
5. **`CC`, `CXX`, and `CUDAHOSTCXX` are shell variables.** They select compilers for tools that honor them. `unset CC CXX CUDAHOSTCXX` removes the overrides and returns compiler selection to defaults; it does not uninstall anything.
6. **`cc1plus` is GCC's C++ backend.** A healthy GCC 8 installation contains it under `"$CONDA_PREFIX/libexec/gcc/x86_64-conda-linux-gnu/8.4.0/cc1plus"`. The direct test below verifies the backend without relying on `which gcc`.
7. **NVCC may ignore `CUDAHOSTCXX` unless `-ccbin` is explicit.** The extension setup now adds `-ccbin` from `CUDAHOSTCXX` or `CXX`, avoiding a wrapper directory and preventing NVCC from falling back to `/usr/bin/gcc`.
8. **RTX 5090 architecture support is not present in the legacy stack.** PyTorch 1.4/CUDA 10.1 cannot identify the GPU's native compute capability. `TORCH_CUDA_ARCH_LIST=7.5` allows compilation as a compatibility experiment, but it does not produce native RTX 5090 kernels. A successful build does not guarantee runtime execution; a later CUDA-kernel test is required.
9. **Extension import order matters.** The generated `.so` files are not standalone Python modules. Load `torch` first, then `deform_conv_cuda` and `deform_pool_cuda`. Importing the extension first can fail even when compilation and linkage are otherwise correct.

### Compiler smoke tests

Run these commands from the activated legacy environment:

```bash
echo "$CC"
echo "$CXX"
echo "$CUDAHOSTCXX"

"$CC" --version
"$CXX" --version
"$CC" -print-prog-name=cc1plus

printf 'int main() { return 0; }\n' | \
  "$CXX" -x c++ -std=c++11 -c -o /tmp/dcpose_cc1plus_test.o -
rm -f /tmp/dcpose_cc1plus_test.o
```

Expected results are GCC/G++ 8.4.0, a real `cc1plus` path, and a successful temporary compilation. A failure here is a compiler installation or environment problem; an error only during NVCC compilation indicates host-compiler selection and should be investigated through the `-ccbin` argument.

### Extension smoke tests

After `python setup.py develop` completes, run:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
import deform_conv_cuda
import deform_pool_cuda
print("deformable convolutions loaded")
PY
```

If this succeeds, follow it with a small actual CUDA operator/model execution. Import success alone does not prove that the RTX 5090 can execute kernels compiled by the legacy CUDA 10.1 stack. If the import fails with `THPVariableClass` or `c10::BFloat16` while `torch` was imported first, investigate PyTorch ABI/build compatibility; do not reinstall GCC as the first response.

## Operational details to keep in mind

- The demo detects people independently in every frame; it does not track person identities.
- The output video writer in [`utils/utils_video.py`](./utils/utils_video.py) defaults to 25 FPS rather than preserving the input FPS.
- The output codec is fixed to `DIVX` and may need to be changed to `mp4v`.
- The default checkpoint is PoseTrack17-specific.
- The legacy deformable-convolution build was validated only as a compatibility experiment on the RTX 5090 with `TORCH_CUDA_ARCH_LIST=7.5`; native architecture support requires the modernization path.
- The `--gpu_id` argument in [`tools/inference.py`](./tools/inference.py) does not by itself control all CUDA usage; the code uses direct `.cuda()` calls and the detector has its own GPU environment setting.
- The official pipeline is separate from the Ultralytics-based scripts in [`../Week4Demo/`](../Week4Demo/).
