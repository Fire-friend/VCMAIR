# VCMAIR

Code for **Virtual Consistency Model for All-in-One Image Restoration**.
Supports deraining, desnowing, dehazing, deblurring, and low-light enhancement.

## Installation

Run commands from the repository root.

```bash
conda create -n vcmair python=3.10 -y
conda activate vcmair
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Inference

Place `vcmair.pth` in `checkpoints/`. This single file includes all inference
weights and configuration, including the VAE. No model download is needed at runtime.

```bash
python infer.py \
  --input input.png \
  --output output.png \
  --model checkpoints/vcmair.pth
```

`--input` also accepts a directory. Defaults: CUDA, float16, one step.
Run `python infer.py --help` for all options.

## Training

Single-stage joint training of the diffusion U-Net LoRA, residual U-Net, and
prompt mapper. Fusion Network training is excluded.

See [training/README.md](training/README.md) for data preparation, training,
and exporting a single-file inference checkpoint.

## Citation

```bibtex
@article{wu2026vcmair,
  author  = {Jiawei Wu and Luwei Tu and Zhe Wang and Zhi Jin and
             Kaihao Zhang and Wenqi Ren and Xiaochun Cao},
  title   = {Virtual Consistency Model for All-in-One Image Restoration},
  journal = {IEEE Transactions on Image Processing},
  volume  = {35},
  pages   = {7506--7520},
  year    = {2026},
  doi     = {10.1109/TIP.2026.3710487}
}
```

## License

Project license: to be added. Bundled backbone weights retain their original licenses.
