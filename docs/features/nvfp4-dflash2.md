# Weight-only NVFP4 DFlash2 runtime

The Qwen3.8-27B DFlash2 module (34 NVFP4 objects) executes weight-only NVFP4 with BF16
activations, not invented A4 calibration. Runtime execution/binding covers attention
projections, context materialization, dynamic convolution, linear/SwiGLU projections and
the candidate-selector path. It preserves tile-aligned NVFP4 subview scale offsets and
admits lookup codebooks without requiring activation-use metadata that conversion does not
produce.

Use a compatible `.ninfer` artifact with a DFlash2 component and select
`--spec dflash2 --draft-tokens 7`; see [DFlash semantics](../maintainer/dflash.md).
Conversion allocation and model-quality qualification remain separate from runtime support.
