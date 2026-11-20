# Resin patches for Muse Glimmer

Patch series on top of upstream Resin `1fe19de12380cd0f5cf9c05bdd06213272217df7`.
There is one patch per semantic commit. `resin-source/` stays a gitignored
upstream checkout.

```bash
git clone https://github.com/ekzhang/resin.git resin-source
git -C resin-source checkout -b muse-glimmer-torch 1fe19de12380cd0f5cf9c05bdd06213272217df7
git -C resin-source am ../resin-patches/*.patch
```

Regenerate after changing the branch (clear old patches first):

```bash
git -C resin-source format-patch 1fe19de..muse-glimmer-torch -o ../resin-patches
```
