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
cd thirdparty/deform_conv
python setup.py develop
cd ../..
```

The build requires a working CUDA toolkit, `nvcc`, and a PyTorch installation compatible with that toolkit.

Optional import check:

```bash
python -c "import deform_conv_cuda; print('deform_conv_cuda loaded')"
```

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
| Python/PyTorch/CUDA version incompatibility | 2 | `python --version && python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"` | Import errors, missing CUDA, or a PyTorch/CUDA combination incompatible with the installed driver/toolkit. | **Pending smoke test.** Record versions and choose either the pinned legacy environment or a matched modern PyTorch/torchvision pair. |
| Required Python packages are missing | 2 | `python -c "import cv2, numpy, PIL, yaml, torch, torchvision, tqdm; print('runtime imports OK')"` | Any `ModuleNotFoundError` or binary import error. | **Pending smoke test.** Install missing packages from the legacy requirements or the selected modern environment specification; record the exact fix. |
| CUDA toolkit/compiler is unavailable for the custom extension | 3 | `nvcc --version` | `nvcc` is missing or reports a toolkit incompatible with the installed PyTorch CUDA build. | **Pending smoke test.** Install/select a compatible CUDA toolkit or document the modern operator replacement. |
| Deformable-convolution extension did not build or cannot load | 3 | `python -c "import deform_conv_cuda; print('deform_conv_cuda loaded')"` | `ModuleNotFoundError`, undefined symbols, or a shared-library loading error. | **Pending smoke test.** Rebuild against the selected PyTorch/CUDA pair, or port to a behaviorally equivalent maintained operator. |
| Supplementary model files are missing or placed incorrectly | 4–5 | `test -f DcPose_supp_files/object_detector/YOLOv3/yolov3.weights && test -f DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth && echo 'model files found'` | Either `test` command fails, or the symlink points to the wrong directory. | **Pending smoke test.** Download the supplementary archive and place/link it at `DCPose_supp_files` with the expected layout. |
| The supplementary directory symlink is broken | 5 | `readlink -f DcPose_supp_files && find -L DcPose_supp_files -maxdepth 4 -type f | sort | sed -n '1,30p'` | The resolved path does not exist or the expected weights are absent. | **Pending smoke test.** Recreate the symlink using an absolute path and verify both required weights. |
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

## Operational details to keep in mind

- The demo detects people independently in every frame; it does not track person identities.
- The output video writer in [`utils/utils_video.py`](./utils/utils_video.py) defaults to 25 FPS rather than preserving the input FPS.
- The output codec is fixed to `DIVX` and may need to be changed to `mp4v`.
- The default checkpoint is PoseTrack17-specific.
- The `--gpu_id` argument in [`tools/inference.py`](./tools/inference.py) does not by itself control all CUDA usage; the code uses direct `.cuda()` calls and the detector has its own GPU environment setting.
- The official pipeline is separate from the Ultralytics-based scripts in [`../Week4Demo/`](../Week4Demo/).
