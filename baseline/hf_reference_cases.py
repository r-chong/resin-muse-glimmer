"""Independent Transformers reference for Resin parity: long, window-crossing,
and image cases. Same pinned image, checkpoint, and volume as hf_reference.py.

For each case this saves the chat-templated input IDs, logits at selected
prompt positions (including the 2,048-token sliding-window boundary), and
greedy decode logits. The image case also saves processor outputs and the
projected vision features.

    modal run baseline/hf_reference_cases.py --steps 16
"""

import hashlib
import json
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
MODEL_ID = "meta-models/Muse-Glimmer-30B"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
CURRENT_DATE = "2026-11-19"  # fixed so the template's date line is reproducible
IMAGE_PATH = ROOT / "fixtures" / "muse_check_644x476.png"
MOUNT = Path("/shared")
VOLUME = modal.Volume.from_name("muse-glimmer-transformers-reference-v1", version=2, create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.9.1",
    "torchvision==0.24.1",
    "transformers==5.19.0",
    "accelerate==1.15.0",
    "huggingface_hub==2.4.0",
    "numpy>=2,<3",
    "pillow==11.3.0",
)
app = modal.App("muse-glimmer-transformers-reference-cases")


def long_text(sentences: int) -> str:
    """Deterministic varied prose: one counting sentence per integer."""
    colors = ["red", "green", "blue", "amber", "violet", "silver", "teal"]
    animals = ["otter", "heron", "lynx", "badger", "falcon", "marmot", "gecko"]
    return " ".join(
        f"Entry {i}: the {colors[i % 7]} {animals[(i * 3) % 7]} counted {i * 17 % 101} stones"
        f" and {'kept' if i % 2 else 'gave away'} {i % 9} of them."
        for i in range(sentences)
    )


def cases() -> list[dict]:
    gsm8k = json.loads((ROOT / "prompt.jsonl").read_text())
    question = "\n\nQuestion: How many stones did entry 37 count? Answer briefly."
    return [
        {"id": gsm8k["id"], "messages": gsm8k["messages"], "probe": []},
        {
            "id": "long_window",
            "messages": [{"role": "user", "content": long_text(190) + question}],
            "probe": [1024, 2046, 2047, 2048, 2049, 2050, 4095, 4096],
        },
        {
            "id": "long_8k",
            "messages": [{"role": "user", "content": long_text(345) + question}],
            "probe": [2047, 4096, 6143, 8000],
        },
        {
            "id": "image_check",
            "messages": [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the shapes, colors, and text in this image."},
            ]}],
            "probe": [],
            "image": True,
        },
    ]


@app.function(image=IMAGE, gpu="H100!", volumes={MOUNT: VOLUME}, cpu=8, memory=65536, timeout=7200)
def capture(all_cases: list[dict], image_bytes: bytes, steps: int, run_id: str) -> dict:
    import importlib.metadata as metadata
    import io
    import platform

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoProcessor, MuseGlimmerForConditionalGeneration

    VOLUME.reload()
    checkpoint = snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=MOUNT / "models" / REVISION)
    VOLUME.commit()
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = processor.tokenizer
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda:0", low_cpu_mem_usage=True, local_files_only=True
    ).eval()
    stop_ids = {tokenizer.convert_tokens_to_ids("<|eot|>"), tokenizer.convert_tokens_to_ids("<|end_of_text|>")}
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    directory = MOUNT / "runs" / run_id
    directory.mkdir(parents=True)
    results = []
    for case in all_cases:
        text = processor.apply_chat_template(
            case["messages"], add_generation_prompt=True, tokenize=False, current_date=CURRENT_DATE
        )
        encoded = processor(
            text=[text], images=[image] if case.get("image") else None, return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to("cuda:0")
        length = input_ids.shape[1]
        extra = {}
        arrays = {}
        if case.get("image"):
            extra["pixel_values"] = encoded["pixel_values"].to("cuda:0", torch.bfloat16)
            extra["image_grid_thw"] = encoded["image_grid_thw"].to("cuda:0")
            arrays["pixel_values"] = encoded["pixel_values"].float().numpy()
            arrays["image_grid_thw"] = encoded["image_grid_thw"].numpy()
            with torch.inference_mode():
                features = model.get_image_features(extra["pixel_values"], extra["image_grid_thw"]).pooler_output
            arrays["image_features"] = torch.cat(features).float().cpu().numpy().astype(np.float16)
        probe = sorted({p for p in case["probe"] if p < length - 1} | {length - 1})
        with torch.inference_mode():
            output = model(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True,
                logits_to_keep=torch.tensor(probe, device="cuda:0"), **extra,
            )
            probe_logits = output.logits[0].float().cpu().numpy()
            cache = output.past_key_values
            mask = torch.ones_like(input_ids)
            token = int(probe_logits[-1].argmax())
            step_logits = [probe_logits[-1]]
            generated = [token]
            for _ in range(steps - 1):
                if token in stop_ids:
                    break
                current = torch.tensor([[token]], device="cuda:0")
                mask = torch.cat((mask, torch.ones_like(current)), dim=1)
                output = model(input_ids=current, attention_mask=mask, past_key_values=cache, use_cache=True,
                               logits_to_keep=1)
                cache = output.past_key_values
                row = output.logits[0, -1].float().cpu().numpy()
                step_logits.append(row)
                token = int(row.argmax())
                generated.append(token)
        if not all(np.isfinite(row).all() for row in step_logits):
            raise RuntimeError(f"non-finite logits in {case['id']}")
        np.savez_compressed(
            directory / f"{case['id']}.npz",
            input_ids=input_ids[0].cpu().numpy().astype(np.int64),
            probe_positions=np.array(probe, dtype=np.int64),
            probe_logits=probe_logits,
            step_logits=np.stack(step_logits),
            generated_ids=np.array(generated, dtype=np.int64),
            **arrays,
        )
        results.append({
            "id": case["id"],
            "messages": case["messages"],
            "file": f"{case['id']}.npz",
            "input_length": length,
            "probe_positions": probe,
            "image_tokens": int((input_ids == model.config.image_token_id).sum()),
            "image_grid_thw": encoded["image_grid_thw"].tolist() if case.get("image") else None,
            "generated_ids": generated,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=False),
            "stopped_on_eos": generated[-1] in stop_ids,
        })
    packages = ["torch", "torchvision", "transformers", "tokenizers", "accelerate", "huggingface_hub",
                "numpy", "pillow", "safetensors"]
    manifest = {
        "format": "muse-transformers-reference-cases-v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "current_date": CURRENT_DATE,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "dtype": "bfloat16",
        "attn_implementation": model.config._attn_implementation,
        "device": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "versions": {name: metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "greedy_steps": steps,
        "cases": results,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    VOLUME.commit()
    return manifest


@app.local_entrypoint()
def main(steps: int = 16) -> None:
    run_id = uuid.uuid4().hex
    manifest = capture.remote(cases(), IMAGE_PATH.read_bytes(), steps, run_id)
    destination = ROOT / "output" / run_id
    destination.mkdir(parents=True)
    for name in [case["file"] for case in manifest["cases"]] + ["manifest.json"]:
        with (destination / name).open("wb") as stream:
            for chunk in VOLUME.read_file(f"runs/{run_id}/{name}"):
                stream.write(chunk)
    for case in manifest["cases"]:
        print(case["id"], case["input_length"], case["image_tokens"], repr(case["generated_text"][:120]))
    print(destination)
