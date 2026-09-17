# RVT: Recurrent Vision Transformers for Object Detection with Event Cameras
<p align="center">
  <img src="https://rpg.ifi.uzh.ch/img/papers/arxiv22_detection_mgehrig/combo.png" width="750">
</p>

This is the official Pytorch implementation of the CVPR 2023 paper [Recurrent Vision Transformers for Object Detection with Event Cameras](https://arxiv.org/abs/2212.05598).

Watch the [**video**](https://youtu.be/xZ-pNwHxHgY) for a quick overview.

```bibtex
@InProceedings{Gehrig_2023_CVPR,
  author  = {Mathias Gehrig and Davide Scaramuzza},
  title   = {Recurrent Vision Transformers for Object Detection with Event Cameras},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2023},
}
```

## Installation
### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the Python 3.9 environment used by
this project. `uv` will download Python 3.9 automatically if it is not already available:

```Bash
uv python install 3.9
uv venv --python 3.9
source .venv/bin/activate
```

On macOS, install the platform-native PyTorch wheels followed by the remaining dependencies:

```Bash
uv pip install -r torch-req.txt
uv pip install -r requirements.txt
```

On Linux with CUDA 11.8, use the PyTorch CUDA wheel index instead:

```Bash
uv pip install -r torch-req.txt --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
```

Detectron2 is optional and only speeds up evaluation. Install it in the active environment if needed:

```Bash
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

The checked-in `.python-version` lets later `uv` commands select Python 3.9 automatically. To reactivate an existing
environment, run `source .venv/bin/activate`.

### Conda
We highly recommend to use [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to reduce the installation time.
```Bash
conda create -y -n rvt python=3.9 pip
conda activate rvt
conda config --set channel_priority flexible

CUDA_VERSION=11.8

conda install -y h5py=3.8.0 blosc-hdf5-plugin=1.0.0 \
hydra-core=1.3.2 einops=0.6.0 torchdata=0.6.0 tqdm numba \
pytorch=2.0.0 torchvision=0.15.0 pytorch-cuda=$CUDA_VERSION \
-c pytorch -c nvidia -c conda-forge

python -m pip install pytorch-lightning==1.8.6 wandb==0.14.0 \
pandas==1.5.3 plotly==5.13.1 opencv-python==4.6.0.66 tabulate==0.9.0 \
pycocotools==2.0.6 bbox-visualizer==0.1.0 StrEnum==0.4.10
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Detectron2 is not strictly required but speeds up the evaluation.

### Venv
Alternative to the conda installation.
```Bash
python -m venv rvt
source rvt/bin/activate
python -m pip install -r torch-req.txt --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```
Optionally, install Detectron2 within the *activated* venv
```Bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Required Data
To evaluate or train RVT you will need to download the required preprocessed datasets:

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
</tr>
</tbody></table>

You may also pre-process the dataset yourself by following the [instructions](scripts/genx/README.md).

## Pre-trained Checkpoints
### 1 Mpx
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">RVT-Base</th>
<th valign="bottom">RVT-Small</th>
<th valign="bottom">RVT-Tiny</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-b.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-s.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-t.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>72923a</tt></td>
<td align="center"><tt>a94207</tt></td>
<td align="center"><tt>5a3c78</tt></td>
</tr>
</tbody></table>

### Gen1
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">RVT-Base</th>
<th valign="bottom">RVT-Small</th>
<th valign="bottom">RVT-Tiny</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-b.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-s.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/gen1/rvt-t.ckpt">download</a></td>
</tr>
<tr><td align="left">md5</td>
<td align="center"><tt>839317</tt></td>
<td align="center"><tt>840f2b</tt></td>
<td align="center"><tt>a770b9</tt></td>
</tr>
</tbody></table>

## Evaluation
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set `CKPT_PATH` to the path of the *correct* checkpoint matching the choice of the model and dataset.
- Set
  - `MDL_CFG=base`, or
  - `MDL_CFG=small`, or
  - `MDL_CFG=tiny`
  
  to load either the base, small, or tiny model configuration
- Set
  - `USE_TEST=1` to evaluate on the test set, or
  - `USE_TEST=0` to evaluate on the validation set
- Set `GPU_ID` to the PCI BUS ID of the GPU that you want to use. e.g. `GPU_ID=0`.
  Only a single GPU is supported for evaluation
### 1 Mpx
```Bash
python validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```
### Gen1
```Bash
python validation.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen1="${MDL_CFG}.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```

## Training
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set
    - `MDL_CFG=base`, or
    - `MDL_CFG=small`, or
    - `MDL_CFG=tiny`

  to load either the base, small, or tiny model configuration
- Set `GPU_IDS` to the PCI BUS IDs of the GPUs that you want to use. e.g. `GPU_IDS=[0,1]` for using GPU 0 and 1.
  **Using a list of IDS will enable single-node multi-GPU training.**
  Pay attention to the batch size which is defined per GPU:
- Set `BATCH_SIZE_PER_GPU` such that the effective batch size is matching the parameters below.
  The **effective batch size** is (batch size per gpu)*(number of GPUs).
- If you would like to change the effective batch size, we found the following learning rate scaling to work well for 
all models on both datasets:
  
  `lr = 2e-4 * sqrt(effective_batch_size/8)`.
- The training code uses [W&B](https://wandb.ai/) for logging during the training.
Hence, we assume that you have a W&B account. 
  - The training script below will create a new project called `RVT`. Adapt the project name and group name if necessary.
 
### 1 Mpx
- The effective batch size for the 1 Mpx training is 24.
- To train on 2 GPUs using 6 workers per GPU for training and 2 workers per GPU for evaluation:
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=12
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen4 dataset.path=${DATA_DIR} wandb.project_name=RVT \
wandb.group_name=1mpx +experiment/gen4="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```
If you instead want to execute the training on 4 GPUs simply adapt `GPU_IDS` and `BATCH_SIZE_PER_GPU` accordingly:
```Bash
GPU_IDS=[0,1,2,3]
BATCH_SIZE_PER_GPU=6
```
### Gen1
- The effective batch size for the Gen1 training is 8.
- To train on 1 GPU using 6 workers for training and 2 workers for evaluation:
```Bash
GPU_IDS=0
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=6
EVAL_WORKERS_PER_GPU=2
python train.py model=rnndet dataset=gen1 dataset.path=${DATA_DIR} wandb.project_name=RVT \
wandb.group_name=gen1 +experiment/gen1="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```

## Visualizing recurrent hidden states

`visualize_h_state.py` runs a pretrained model over one preprocessed sequence and writes an MP4 containing the event
input and the ConvLSTM hidden state (`h`) from each recurrent backbone stage. No training is performed. Set
`DATA_DIR`, `CKPT_PATH`, and `MDL_CFG` as in the evaluation examples, then run:

```Bash
python visualize_h_state.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
+experiment/gen1="${MDL_CFG}.yaml" visualization.output=h_state.mp4
```

The first sequence in the validation split is used by default. Select a sequence by directory name or index, limit
the output while testing, or change the device as follows:

```Bash
python visualize_h_state.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
+experiment/gen4="${MDL_CFG}.yaml" visualization.sequence=moorea_2019-02-19_000_0_0 \
visualization.max_frames=300 visualization.output=outputs/moorea_h_state.mp4

# CPU inference (considerably slower)
python visualize_h_state.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
+experiment/gen1="${MDL_CFG}.yaml" visualization.device=cpu visualization.sequence_index=2
```

By default, channels are reduced with their mean absolute activation and each stage maintains its own robust,
temporally smoothed color scale. `visualization.channel_reduction=mean` preserves the activation sign instead.
Other useful overrides are `visualization.stages=[1,2,3,4]`, `visualization.fps=20`,
`visualization.percentile=99`, and `visualization.scale_ema_decay=0.95`.

The MP4 header shows the sequence name and a zero-based frame number. Once a useful frame has been identified, rerun
in image-only mode to export the event image and every selected hidden-state stage as separate, native-resolution files:

```Bash
python visualize_h_state.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
+experiment/gen1="${MDL_CFG}.yaml" visualization.sequence=SEQ_NAME visualization.write_video=false \
visualization.image_export.enable=true 'visualization.image_export.frames=[120,245]' \
visualization.image_export.output_dir=outputs/paper_frames visualization.image_export.format=png
```

Exported images have no embedded title by default, making them suitable for figures. Their filenames and
`<sequence>_metadata.json` retain the sequence, frame, stage, reduction, shape, and color-scale information. Set
`visualization.image_export.include_titles=true` if labeled standalone images are preferred. JPEG output is available
with `visualization.image_export.format=jpg`.

## JetPilot EVS RAW / ROS bag inference

`visualize_h_state_raw.py` accepts event recordings produced by JetPilot and runs the complete RVT backbone and
detection head while visualizing the recurrent hidden states. Supported inputs are:

| Input | Handling |
|---|---|
| JetPilot `EVSBIN v1` (`.evbin`) | Read directly with a disk-backed memory map |
| OpenEB native recording (`.raw`) | Converted with JetPilot's `evs_raw_to_evbin` tool |
| ROS 2 bag (`.mcap`, `.db3`, or bag directory) | `EventPacket` is decoded and converted to EVSBIN by an isolated helper process |

The adapter reproduces the pretrained RVT input contract: a 50 ms window and stride, 10 temporal bins, separate
polarity-major channels, and a per-pixel/bin count cutoff of 10. JetPilot's default online tensor uses a different
40 ms window, 4 ms stride, and opposite polarity ordering, so it must not be passed to RVT without this adaptation.
The source geometry is center-cropped and resized in event-coordinate space by default; use
`source.geometry_mode=letterbox` to preserve the complete field of view or `stretch` to fill the frame.

### EVSBIN input

```Bash
python visualize_h_state_raw.py dataset=gen4 checkpoint=${CKPT_PATH} \
+experiment/gen4="${MDL_CFG}.yaml" source.path=/path/to/events.evbin \
visualization.output=outputs/events_rvt.mp4
```

The video contains six panels by default: events, RVT detections, and hidden states from stages 1–4. Its header shows
the source name, zero-based frame number, relative time, and event count.

### Native OpenEB RAW input

First build `evs_raw_to_evbin` in JetPilot's `tools/evs_benchmark` project with the Metavision SDK available. Pass the
resulting executable to RVT:

```Bash
python visualize_h_state_raw.py dataset=gen4 checkpoint=${CKPT_PATH} \
+experiment/gen4="${MDL_CFG}.yaml" source.path=/path/to/recording.raw \
source.raw_converter=/path/to/JetPilot/tools/evs_benchmark/build/evs_raw_to_evbin \
visualization.output=outputs/recording_rvt.mp4
```

The reordered canonical file is cached next to the MP4 as `<source>.rvt.evbin`. Set `source.reuse_cache=false` to
reconvert it or set `source.evbin_cache=/path/to/cache.evbin` to choose an explicit location.

### ROS 2 bag input

Bag conversion needs `rosbags` and JetPilot/OpenEB's `event_camera_py`. Because `event_camera_py` is normally built
for the ROS system Python while RVT uses Python 3.9, decoding runs in a separate process. Source the JetPilot ROS
workspace first and point `source.rosbag_python` at the interpreter that can import `event_camera_py`:

```Bash
python visualize_h_state_raw.py dataset=gen4 checkpoint=${CKPT_PATH} \
+experiment/gen4="${MDL_CFG}.yaml" source.path=/path/to/rosbag_directory \
source.event_topic=/event_camera/events source.rosbag_python=/usr/bin/python3 \
visualization.output=outputs/bag_rvt.mp4
```

If `rosbags` is not already installed in that decoder environment, install `raw-input-req.txt` there. The default
topic is `/event_camera/events`; `/event_camera/events_raw` can be selected with `source.event_topic`.

### Selecting a segment and exporting paper figures

Offsets and duration are in milliseconds. In image-only mode, the selected event image, detection image, and every
hidden-state stage are written separately at their native visualization resolutions:

```Bash
python visualize_h_state_raw.py dataset=gen4 checkpoint=${CKPT_PATH} \
+experiment/gen4="${MDL_CFG}.yaml" source.path=/path/to/events.evbin \
source.start_offset_ms=12000 source.duration_ms=5000 visualization.write_video=false \
visualization.image_export.enable=true 'visualization.image_export.frames=[20,42]' \
visualization.image_export.output_dir=outputs/paper_frames
```

The recurrent state is always computed from the beginning of the selected segment up to each requested frame. The
export metadata records source and model geometry, representation parameters, timestamps, event counts, stage shapes,
and color scales.

The released RVT checkpoints were trained on Prophesee Gen1/1 Mpx data rather than SilkyEvCam recordings. The adapter
makes the tensor contract compatible, but it does not remove sensor/domain shift; detection accuracy must therefore be
validated quantitatively before the predictions are used as research results.

## Works Built on This Project
- [LEOD: Label-Efficient Object Detection for Event Cameras](https://github.com/Wuziyi616/LEOD). CVPR 2024
- [State Space Models for Event Cameras](https://github.com/uzh-rpg/ssms_event_cameras). CVPR 2024

Open a pull request if you would like to add your project here.

## Code Acknowledgments
This project has used code from the following projects:
- [timm](https://github.com/huggingface/pytorch-image-models) for the MaxViT layer implementation in Pytorch
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) for the detection PAFPN/head
