# Visual-RFT-origin

Visual-RFT-origin is a reinforcement-learning project for farmland visual perception and reasoning. It uses GRPO to optimize a vision-language model (VLM), LoRA for parameter-efficient training, rule-based rewards for structured predictions, and a teacher model to evaluate the generated reasoning process.

The farmland training entry point is:

```text
src/virft/src/open_r1/farmland_percept_reason.py
```

## Installation

The environment is based on the official [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT) project. Linux, CUDA-compatible GPUs, and Python 3.10 are recommended.

```bash
cd Visual-RFT-origin

conda create -n visual-rft python=3.10 -y
conda activate visual-rft

cd src/virft
pip install -e ".[dev]"
cd ../..

bash setup.sh
pip install flask requests scipy opencv-python-headless
pip install flash-attn --no-build-isolation
```

Install a CUDA-enabled PyTorch build compatible with your NVIDIA driver. FlashAttention must also match the installed CUDA and PyTorch versions.

## Model Preparation

Prepare a policy VLM and a teacher reward model:

```text
checkpoints/
├── Qwen2.5-VL-7B-Instruct/   # policy model
└── Qwen3-32B/                # teacher reward model
```

- Set `base_model` in `farmland_percept_reason.py` to the Qwen2.5-VL checkpoint.
- Set the teacher model path in `qwen3.py`.
- Change `.to("cuda:2")` in `qwen3.py` if the teacher model should use another GPU.

`eval_model` is currently a reserved argument. The teacher model is loaded directly by `qwen3.py`.

## Dataset Preparation

`data_path` must point to a local Hugging Face dataset saved with `DatasetDict.save_to_disk()`:

```text
HFdata/
└── DatasetDict
    └── train: Dataset
        ├── image
        ├── problem
        └── solution
```

Each training sample contains:

- `image`: an iterable collection of images. Use a one-element collection for a single-image sample.
- `problem`: the visual question or instruction.
- `solution`: the reference result used by the reward functions.

Example:

```python
from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_dict(
    {
        "image": [["/data/images/example.tif"]],
        "problem": ["<image>\nIdentify and describe the farmland region."],
        "solution": ["Reference answer"],
    }
)

DatasetDict({"train": train_dataset}).save_to_disk("/data/HFdata")
```

The training entry uses `load_from_disk()`, so `data_path` must be a saved local dataset directory rather than a Hugging Face Hub dataset ID.

## Configuration

Edit the defaults in `parse_args()`:

```text
src/virft/src/open_r1/farmland_percept_reason.py
```

| Group | Parameter | Default | Description |
| --- | --- | --- | --- |
| Path | `base_model` | `""` | Policy VLM checkpoint |
| Path | `eval_model` | `""` | Reserved teacher-model argument |
| Path | `data_path` | `""` | Local `DatasetDict` directory |
| Path | `model_save_path` | `""` | LoRA output directory |
| Training | `epochs` | `1` | Number of training epochs |
| Training | `batch_size` | `1` | Per-device batch size |
| Training | `gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| Training | `learning_rate` | `5e-6` | Learning rate |
| Reserved | `beta1` | `0.9` | Parsed but not applied by the current entry point |
| Reserved | `beta2` | `0.95` | Parsed but not applied by the current entry point |
| Reserved | `precision` | `"bf16"` | Parsed but not applied by the current entry point |
| GRPO | `KL_penaly` | `0.05` | KL penalty coefficient |
| GRPO | `num_generations` | `6` | Responses generated for each prompt |
| GRPO | `temperature` | `1` | Sampling temperature |
| GRPO | `top_p` | `0.9` | Nucleus sampling threshold |
| GRPO | `top_k` | `None` | Top-k sampling threshold |
| GRPO | `max_prompt_length` | `2048` | Maximum prompt length |
| GRPO | `max_completion_length` | `1024` | Maximum generated-response length |
| LoRA | `use_peft` | `True` | Enable PEFT/LoRA |
| LoRA | `lora_r` | `16` | LoRA rank |
| LoRA | `lora_alpha` | `32` | LoRA scaling factor |
| LoRA | `lora_dropout` | `0.05` | LoRA dropout |
| LoRA | `lora_target_modules` | See below | Modules receiving LoRA adapters |

Default LoRA target modules:

```python
[
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

## Training

Before training, update the local paths, Conda environment, GPU IDs, and process count in:

```text
Visual-RFT-origin/src/scripts/train_RL.sh
```

Start the teacher reward service and LoRA-based GRPO training with:

```bash
bash Visual-RFT-origin/src/scripts/train_RL.sh
```

The script starts `qwen3.py` in the background and then launches `farmland_percept_reason.py` with `torchrun`. The reward service must remain available at `http://127.0.0.1:7891/chat`.

Ensure that `NPROC_PER_NODE` matches the number of training GPUs in `GPUS`. Assign the teacher model to a separate GPU when possible.

## Output

The trained LoRA artifacts are saved to the directory configured by `model_save_path`.

## Memory Notes

If GPU memory is insufficient, reduce `num_generations`, `batch_size`, image resolution, `max_prompt_length`, or `max_completion_length`. Gradient accumulation can be increased to maintain the effective batch size.

## Acknowledgement

This project is built upon [Liuziyu77/Visual-RFT](https://github.com/Liuziyu77/Visual-RFT).
