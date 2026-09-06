# Training

Run all commands from the repository root after installing the inference dependencies.

```bash
pip install -r requirements-training.txt
# Optional optimizations
pip install bitsandbytes==0.44.1 xformers==0.0.28.post1
```

## Data

```text
DATA_ROOT/
├── RESIDE/OTS_ALPHA/
│   ├── haze/OTS/
│   └── clear/clear_images/
├── RNI15/
├── syn_rain/train/
│   ├── input/
│   └── target/
├── Snow100K/train/
│   ├── synthetic/
│   └── gt/
└── Deblur/train/
    ├── input/
    └── target/
```

Pairs use matching filename stems; RESIDE uses the hazy filename prefix before
`_`. By default, RNI15 uses the same image as input and target. To use paired
low-light data, add `--light-input-dir /path/to/low --light-target-dir /path/to/high`.

## Train

Jointly train LoRA, the residual U-Net, and the prompt mapper in one stage:

```bash
accelerate launch --multi_gpu --num_processes 4 training/train.py \
  --data-root /path/to/DATA_ROOT \
  --output-dir training_outputs \
  --max-train-steps 29000 \
  --mixed-precision fp16
```

Use `--base-model` and `--clip-model` for local backbone paths. Training also
requires LPIPS/VGG weights. Fusion Network training is not included.

Add `--resume-from-checkpoint latest` to resume an interrupted run. Checkpoints
are saved every 500 steps. Run `python training/train.py --help` for options.

Default per-GPU task batches are `8,1,4,4,2` for fog, low-light, rain, snow, and
blur. Training at 256×256 can exceed 24 GB VRAM even with one image per task.
For RTX 40-series NCCL errors, set `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`.

## Export for inference

Update the main-model weights while retaining the bundle's fixed Fusion weights:

```bash
python tools/update_bundle.py \
  --base-bundle checkpoints/vcmair.pth \
  --main-checkpoint training_outputs/checkpoint-29000 \
  --output checkpoints/vcmair_trained.pth
```

Pass the resulting file to `infer.py --model`.
