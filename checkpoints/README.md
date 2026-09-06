# VCMAIR checkpoint

Inference requires exactly one model file:

```text
vcmair.pth
```

It contains the diffusion U-Net/LoRA, VAE, CLIP vision encoder, residual U-Net,
prompt mapper, fusion network, fixed prompt embeddings, and model configuration.
No Hugging Face model or additional project checkpoint is loaded at runtime.

Verify the file with:

```bash
sha256sum -c SHA256SUMS
```
