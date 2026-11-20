# Resin patches for Muse Glimmer

A patch series against upstream Resin
`1fe19de12380cd0f5cf9c05bdd06213272217df7`, split into two groups:

| Directory | Contents |
| --- | --- |
| [`infra/`](infra/) | Resin infrastructure fixes that any new model of this kind would need. They are not Muse-specific. |
| [`model/`](model/) | The Muse Glimmer integration: recipe, bindings, runner, registry entry, and parity test. |

`resin-source/` is a gitignored working checkout on branch
`muse-glimmer-torch`. These patches are the record.

## Apply

```bash
scripts/apply_resin_patches.sh            # clones upstream into ./resin-source
scripts/apply_resin_patches.sh /tmp/resin # or into any new directory
```

Verified on 2026-11-19: both groups apply cleanly to `1fe19de`. The resulting
tree (`9201163761a58d551b7ccf24a30addd3405e5ad9`) is identical to the working
fork.

## Regenerate after changing the fork

Keep infrastructure commits before model commits on the branch, then:

```bash
rm -f resin-patches/infra/*.patch resin-patches/model/*.patch
git -C resin-source format-patch 1fe19de..<last-infra-commit> -o ../resin-patches/infra
git -C resin-source format-patch <last-infra-commit>..HEAD -o ../resin-patches/model
```

## Size (added lines)

| Patch | Implementation | Tests | Fixtures |
| --- | ---: | ---: | ---: |
| infra 0001 ship torch runners in Modal image | 8 | | |
| model 0001 Rust ATEM recipe and image preprocessing | 903¹ | 130 | 287 |
| model 0002 Python bindings (recipe factory, media payload, stubs) | 161 | | |
| model 0003 PyTorch runner (config, text model, vision, Engine runner) | 1,221 | | |
| model 0004 registry preset | 59 | | |
| model 0005 HF parity test | | 111 | |

¹ Includes in-file unit tests: a 514-case `smart_resize` table on one line,
and recipe and media tests.

Files on the **text-only** path: `config.py` (176), `model.py` (448),
`reference.py` (308), `muse.rs` recipe (616), and the registry entry (59).
Image-only code, which is not yet validated on GPU: `vision.py` (274),
`media/muse.rs` (283), and the `MuseGlimmerMedia` binding.
