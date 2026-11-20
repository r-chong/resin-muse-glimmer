# Resin patches for Muse Glimmer

A patch series against upstream Resin
`1fe19de12380cd0f5cf9c05bdd06213272217df7`, split into two groups:

| Directory | Contents |
| --- | --- |
| [`infra/`](infra/) | Resin infrastructure fixes that any new model of this kind would need. They are not Muse-specific. |
| [`model/`](model/) | The Muse Glimmer integration: recipe, bindings, runner, registry entry, and parity test. |

`resin-source/` is **vendored** in this repo. Commit `a1736eb` is pristine
upstream `1fe19de`. Each later commit touching `resin-source/` is one fork
change, with infrastructure commits first. These patches are generated from
that history.

## Apply

```bash
scripts/apply_resin_patches.sh /tmp/resin   # fresh upstream clone + both groups
```

Verified on 2026-11-19: both groups apply cleanly to `1fe19de`. The resulting
tree (`f2b6cab854f077b0ac5e7f6f9d23bb943d4444f8`) is identical to the vendored
`resin-source/`.

## Regenerate after changing the fork

Edit `resin-source/` directly. Commit infrastructure fixes and model changes
separately, using Conventional Commit messages. Then:

```bash
rm -f resin-patches/infra/*.patch resin-patches/model/*.patch
git format-patch --relative=resin-source a1736eb..<last-infra-commit> -o resin-patches/infra -- resin-source
git format-patch --relative=resin-source <last-infra-commit>..<last-model-commit> -o resin-patches/model -- resin-source
```

If a new infrastructure fix lands after model commits, list it separately
(for example `infra/0002-…`) and note its position. Do not rewrite pushed history.

## Size (added lines)

| Patch | Implementation | Tests | Fixtures |
| --- | ---: | ---: | ---: |
| infra 0001 ship torch runners in Modal image | 8 | | |
| infra 0002 bench supports runners without CUDA-runner extras (after model commits in history) | ~10 | | |
| model 0001 Rust ATEM recipe and image preprocessing | 903¹ | 130 | 287 |
| model 0002 Python bindings (recipe factory, media payload, stubs) | 161 | | |
| model 0003 PyTorch runner (config, text model, vision, Engine runner) | 1,221 | | |
| model 0004 registry preset | 59 | | |
| model 0005 HF parity test | | 111 | |
| model 0006–0007 window/batching tests, logit dump | | ~170 | |
| model 0008 Engine-managed paged KV (Resin `CacheLayout`) | ~345 | 167 | |
| model 0009 batch-width check independent of KV path | | ~20 | |
| model 0010 `kv_cache` spec field, local-KV comparison preset | 35 | | |

¹ Includes in-file unit tests: a 514-case `smart_resize` table on one line,
and recipe and media tests.

Files on the **text-only** path: `config.py` (176), `model.py` (448),
`reference.py` (308), `muse.rs` recipe (616), and the registry entry (59).
Image-only code, which is not yet validated on GPU: `vision.py` (274),
`media/muse.rs` (283), and the `MuseGlimmerMedia` binding.
