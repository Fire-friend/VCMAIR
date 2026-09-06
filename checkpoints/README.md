# VCMAIR checkpoint

Inference requires exactly one model file:

```text
vcmair.pth
```

Download: [Baidu Netdisk](https://pan.baidu.com/s/13ERW6_YolW1dLj_kKlkv7w?pwd=hy2t) (extraction code: `hy2t`).
Save the downloaded file as `checkpoints/vcmair.pth`.

It contains the diffusion U-Net/LoRA, VAE, CLIP vision encoder, residual U-Net,
prompt mapper, fusion network, fixed prompt embeddings, and model configuration.
No Hugging Face model or additional project checkpoint is loaded at runtime.

Verify the file from the `checkpoints/` directory with:

```bash
sha256sum -c SHA256SUMS
```
