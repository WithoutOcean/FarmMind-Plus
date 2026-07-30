# Think with Extra-Image: A Dynamic Farmland Segmentation Agent Driven by  Spatio-Temporal Information Gain

Existing farmland remote sensing image (FRSI) segmentation follows a “Think with Intra-Image” paradigm, assuming that the current image contains sufficient visual evidence for reliable identification. Yet farmland appearance varies with phenology and spatial context and is often confused with other land-cover types, making instantaneous, local observations inadequate. Thus, segmentation ambiguity stems not only from limited model representation, but more fundamentally from the required spatio-temporal information lying beyond the current image. Based on this insight, we redefine FRSI segmentation from an information bottleneck perspective as a dynamic decision process driven by task-relevant extra spatiotemporal information gain. We further propose FarmMind+, a dynamic FRSI segmentation agent that identifies ambiguous regions, reasons about their causes, and queries extra spatiotemporal information on demand for accurate segmentation. To evaluate FarmMind+, we construct GSFS-Bench, the first global-scale, high-resolution FRSI segmentation benchmark that supports reasoning-querying. Experiments show that FarmMind+ achieves more stable segmentation performance than existing methods.

## Overview

- `FarmSeeker_offline.py` 1. Generate the initial segmentation mask; 2. Receive the corrected mask; 3. Indicator Evaluation
- `module/FarmSeeker_interface_offline.py` Receive the initial segmentation mask and initiates the reasoning and retrieval workflow.
- `module/FarmSeeker_Retrieve_reason_offline.py` 1. Detects ambiguous regions; 2. Selects external imagery; 3. Performs multi-image reasoning; 4. Refines the segmentation mask using SAM 2.1.
- `module/Retrieve_data_offline.py` Retrieves and crops the requested multi-temporal or enlarged TIFF imagery.
- `Visual-RFT` Provides the reinforcement fine-tuning framework used to train and optimize FarmSeeker.

The segmentation and correction programs use separate Python environments and communicate through `http service`.
## Installation
FarmSeeker requires Linux, Python 3.10, CUDA, and NVIDIA GPU.

### Environment 1: query and correction

This environment runs `module/FarmSeeker_interface_offline.py`.

```bash
conda create -n farmmind-interface python=3.10 -y
conda activate farmmind-interface
python -m pip install -r requirements_interface.txt
```

### Environment 2: Segmentation

Use a separate environment for `FarmSeeker_offline.py`. Follow the Segearth-R1 environmental installation procedure.
## Training

Training uses Visual-RFT.

### GPU configuration

Training uses four NVIDIA A800 GPUs:

- GPU 0-1: distributed policy-model training with `torchrun`
- GPU 2-3: teacher reward model served by `qwen3.py`


### Models

```text
checkpoints/
├── Qwen2.5-VL-7B-Instruct/   # policy model
└── Qwen3-32B/                # teacher reward model
```

Set `base_model` in `farmland_percept_reason.py` to the Qwen2.5-VL checkpoint and set the teacher path in `qwen3.py`. Change `.to("cuda:2")` in `qwen3.py` when using another GPU.

### Dataset
The dataset will be released upon acceptance.
`data_path` must point to a local Hugging Face `DatasetDict` saved with `save_to_disk()`. Each training sample contains an iterable `image` collection, a `problem`, and a reference `solution`.

```python
from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_dict(
    {
        "image": [["/data/images/example.tif"]],
        "problem": ["<image>\Instruction"],
        "solution": ["Reference answer"],
    }
)
DatasetDict({"train": train_dataset}).save_to_disk("/data/HFdata")
```

The training entry uses `load_from_disk()`, so a Hugging Face Hub dataset ID is not supported.

### Configuration

Edit `parse_args()` in `src/virft/src/open_r1/farmland_percept_reason.py`.

```text
Paths:    base_model="", eval_model="", data_path="", model_save_path=""
Training: epochs=1, batch_size=1, gradient_accumulation_steps=1, learning_rate=5e-6
Reserved: beta1=0.9, beta2=0.95, precision="bf16"
GRPO:     KL_penaly=0.05, num_generations=6, temperature=1, top_p=0.9,
          top_k=None, max_prompt_length=2048, max_completion_length=1024
LoRA:     use_peft=True, lora_r=16, lora_alpha=32, lora_dropout=0.05
Targets:  q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj
```

```bash
bash Visual-RFT/src/scripts/train_RL.sh
```

## Inference

### GPU configuration

Inference uses one NVIDIA A800 GPU. Both the correction service and the initial segmentation process run on GPU 0.

### Prepare Model Weights

```text
checkpoint/
├── Reason_Model/             # complete Qwen2.5-VL checkpoint
├── SAM2/
│   └── sam2.1_hiera_large.pt
└── SegEarth-R1/              # complete SegEarth-R1 checkpoint
```

### Prepare Test Data
The dataset will be released upon acceptance.

```text
dataset/
├── enlarge_dir/     # enlarger spatial-context TIFF images
├── images/          # georeferenced TIFF patches
├── labels/          # binary PNG masks: 0 background, 1 farmland
└── temporal_dir/    # TIFF images from other months
```

Input, enlarger, and temporal images must have compatible CRS and affine-transform metadata. Label names must match their input stems.

Every TIFF filename must include the acquisition month:

```text
Input:    <region>_<map-sheet>_<longitude>_<latitude>_<month>_patch_<split-id>.tif
Example:  NRW_N32G091022_7.3_52.23_05_patch_1024_5632.tif
Label:    dataset/labels/NRW_N32G091022_7.3_52.23_05_patch_1024_5632.png
Context:  dataset/enlarge_dir/NRW_N32G091022_7.3_52.23_05.tif
Temporal: dataset/temporal_dir/NRW_N32G091022_7.3_52.23_07.tif
          dataset/temporal_dir/NRW_N32G091022_7.3_52.23_09.tif
```

### Configuration

Set absolute paths in `module/config/config.yaml`:

```yaml
perception_reason_seg:
  sam_checkpoint: "/path/to/FarmSeeker/module/SAM2/sam2.1_hiera_large.pt"
  sam_cfg: "configs/sam2.1/sam2.1_hiera_l.yaml"
  database_temporal: "/path/to/FarmSeeker/dataset/temporal_dir"
  database_enlarge: "/path/to/FarmSeeker/dataset/enlarge_dir"
  crop_temporal_dir: "/path/to/FarmSeeker/outputs/crop_temporal"
  crop_enlarge_dir: "/path/to/FarmSeeker/outputs/crop_context"
  model_path: "/path/to/Reason_Model"
  device: "cuda"
  data_type: "torch.bfloat16"
  save_path: "/path/to/FarmSeeker/outputs/correct_label"
  threshold: 0.3
```

Keep the shown `sam_cfg` for Hiera Large. The current implementation expects `device: "cuda"`. `data_type` supports `torch.bfloat16`, `torch.float16`, or `torch.float32`; `threshold` is the ground-truth farmland-ratio threshold used for evaluation.

```bash
mkdir -p outputs/crop_temporal outputs/crop_context outputs/correct_label
```

### Start

Start the correction service first. Flask listens on port after model loading.

```bash
# Terminal 1: query and correction
cd /path/to/FarmSeeker/module
conda activate farmmind-interface
CUDA_VISIBLE_DEVICES=0 python FarmSeeker_interface_offline.py
```

Then run SegEarth-R1 in the second environment:

```bash
# Terminal 2: initial segmentation
cd /path/to/FarmSeeker
conda activate farmmind-segearth
export PYTHONPATH="$(pwd):$(pwd)/model:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=0 python FarmSeeker_offline.py \
  --base_seg "/path/to/FarmSeeker/checkpoint/SegEarth-R1" \
  --base_dir "/path/to/FarmSeeker/dataset" \
  --url "http service" \
  --metrics_path "/path/to/FarmSeeker/outputs/metrics.json" \
  --precision bf16
```


