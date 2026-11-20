# Muse Glimmer 30B pins

Recorded 2026-11-19. Copies of the small metadata files from the pinned model
revision are in [`cec94fc/`](cec94fc/), with SHA-256 hashes in
[`cec94fc/SHA256SUMS`](cec94fc/SHA256SUMS). Nothing here came from a GPU run.

## Model revision

- Repo: `meta-models/Muse-Glimmer-30B`
- Revision: `cec94fcfd6512f2cfe708138c32533b346ea4f69`
- Weights: `model-00001-of-00002.safetensors` (49,950,112,952 B, LFS sha256
  `8eef6153…72694`), `model-00002-of-00002.safetensors` (9,603,322,320 B, LFS
  sha256 `b58cc214…4be14381`). Total 59.55 GB, BF16.

## Tokenizer

- `tokenizer.json`: 28,129,897 B, sha256
  `c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d`. BPE with a
  regex split pre-tokenizer; vocab size 202,048 (`config.text_config.vocab_size`).
- `tokenizer_config.json`: sha256 `781e6c74…a9c5`, `tokenizer_class`
  `TokenizersBackend`, `processor_class` `MuseGlimmerProcessor`,
  `model_max_length` 131,072.
- Special token IDs:

| Token | ID | Role |
| --- | --- | --- |
| `<\|begin_of_text\|>` | 200000 | BOS |
| `<\|end_of_text\|>` | 200001 | EOS (generation_config) |
| `<\|eom\|>` | 200007 | ends a message but not the turn (e.g. reasoning) |
| `<\|eot\|>` | 200008 | ends the turn; EOS (generation_config) |
| `<\|finetune_right_pad\|>` | 200018 | pad |
| `<\|start\|>` | 200022 | message header |
| `<\|message\|>` | 200023 | header/body separator |
| `<\|video\|>` | 200091 | video placeholder |
| `<\|patch\|>` | 200092 | image placeholder (`image_token_id`) |

- Stop set: `generation_config.eos_token_id = [200001, 200008]`. This matches the
  stop set in `baseline/hf_reference.py`.

## Chat template

- `chat_template.jinja`: 9,992 B, sha256
  `cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678`.
- "ATEM" format: `<|start|>{role}[ to={recipient}]<|message|>{content}<|eot|>`.
  Assistant reasoning goes to `to=self` and ends with `<|eom|>`. The final answer
  goes to `to=user`. Tool calls use `<atem:function_calls>` XML.
- A system block is always rendered. When no system message is supplied, it
  includes `Reasoning strength: high.` unless the `reasoning_strength` template
  variable is set.
- `add_generation_prompt=True` ends the prompt with `<|start|>assistant` and no
  recipient, so the model picks its channel itself. The saved HF runs begin with
  ` to=self<|message|>…` (token 328, 19669, 200023), which is reasoning.
- Images render as one `<|patch|>` per image in the template. The processor then
  expands it to the visual-token count.
- `tokenizer_config.response_template` defines the parser: content `to=user`,
  reasoning `to=self`, tool calls `<atem:invoke …>`.

## Generation defaults (`generation_config.json`)

`do_sample=true`, `temperature=1.0`, `top_p=0.95`, `top_k=64`, `max_length=131072`.
The HF reference and planned correctness checks use greedy decoding instead.

## Image processor (`processor_config.json`)

`MuseGlimmerImageProcessor` (Transformers v5.19.0 source): patch 14, temporal
patch 2 (each image is duplicated across both temporal slots), merge size 2, mean/std 0.5,
rescale 1/255, resample 1 (PIL LANCZOS, antialias), `max_image_tokens` 4096.
`smart_resize` chooses a grid with the closest aspect ratio, in 28-pixel units,
under the token cap. The processor expands each `<|patch|>` to
`<|image_start|>` + `<|patch|>` × (grid_h·grid_w/4) + `<|image_end|>`. Vision tower
has 50 layers (window and full attention), hidden size 1536. Adapter projects
6144 → 4096 → 6656.

## Text decoder facts that affect the engine

- 52 layers: 39 sliding-window layers (window 2048, RoPE θ=5e5) and 13 full-attention layers
  (every 4th, **NoPE**).
- 32 query heads, 2 KV heads, head dim 128. QK RMSNorm without scale, query × 3.87,
  scale `head_dim**-0.5`. Sigmoid output gate (`gate_proj`) before `o_proj`.
- Four "centered" RMSNorms per layer (`1 + w`), post-norm eps 1e-8. A weightless
  RMSNorm runs on token embeddings and on projected image features.
- Logits are `20 * tanh(lm_head(x) * 0.19611613513818404 / 20)`. Embeddings are not tied.
- KV per token (BF16): 13 full layers × 1 KiB = 13 KiB unbounded. 39 sliding layers ×
  1 KiB = 39 KiB, but capped at the 2,048-token window. At 8,448 positions this is
  about 112 MB full plus 82 MB sliding per request.

## Dependency versions

| Component | Transformers reference (`baseline/hf_reference.py`) | Resin @ `1fe19de` (`uv.lock`) |
| --- | --- | --- |
| Python | 3.11 (image) | >=3.11 |
| torch | 2.9.1 (+cu128 recorded) | 2.13.0 |
| transformers | 5.19.0 | **5.14.1** |
| tokenizers | not recorded | 0.22.2 |
| huggingface_hub | 2.4.0 | 1.27.0 |
| accelerate | 1.15.0 | — |
| numpy | >=2,<3 (resolved version not recorded) | 2.4.6 / 2.5.1 |
| flashinfer-python | — | 0.7.1 |
| triton | — | 3.7.1 |
| nvidia-cutlass-dsl | — | 4.6.2 |
| torchvision | — | 0.28.0 |
| safetensors | — | 0.8.0 |
| pillow | — | 12.3.0 |
| modal | 1.2.2 (local CLI) | 1.5.3 |

Muse Glimmer was added to Transformers in v5.15.0. Its `models/muse_glimmer`
directory is absent at tag v5.14.1 and present at v5.15.0. **Resin's locked
Transformers 5.14.1 therefore cannot load the Muse config, processor, or
reference modules.** The checkpoint `config.json` says it was written by
`5.15.0.dev0`.

The local Modal CLI is 1.2.2, but Resin requires `modal>=1.5.3`. `uv run`
inside `resin-source` uses the locked 1.5.3.

The table lists declared and locked versions. Installed versions inside the
Modal images still need to be captured on the first authorized run.
