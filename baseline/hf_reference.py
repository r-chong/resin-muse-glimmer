import json
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
MODEL_ID = "meta-models/Muse-Glimmer-30B"
REVISION = "cec94fcfd6512f2cfe708138c32533b346ea4f69"
MOUNT = Path("/shared")
VOLUME = modal.Volume.from_name("muse-glimmer-transformers-reference-v1", version=2, create_if_missing=True)
IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.9.1",
    "transformers==5.19.0",
    "accelerate==1.15.0",
    "huggingface_hub==2.4.0",
    "numpy>=2,<3",
)
app = modal.App("muse-glimmer-transformers-reference")


@app.function(image=IMAGE, gpu="H100!", volumes={MOUNT: VOLUME}, cpu=8, memory=32768, timeout=7200)
def capture(case: dict, max_new_tokens: int, run_id: str) -> None:
    """save tokens and logits."""
    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, MuseGlimmerForConditionalGeneration

    # Download if and only if not cached
    VOLUME.reload()
    checkpoint = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=MOUNT / "models" / REVISION,
    )
    VOLUME.commit()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    stop_ids = {
        tokenizer.convert_tokens_to_ids("<|eot|>"),
        tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
    }
    encoded = tokenizer.apply_chat_template(
        case["messages"], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    current_ids = input_ids.to("cuda:0")
    attention_mask = torch.ones_like(current_ids)
    cache = None
    logits = []
    generated_ids = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            kwargs = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "return_dict": True,
                "logits_to_keep": 1,
            }
            if cache is not None:
                kwargs["past_key_values"] = cache
            output = model(**kwargs)
            cache = output.past_key_values
            row = output.logits[0, -1].float().cpu().numpy().copy()
            if not np.isfinite(row).all():
                raise RuntimeError("model returned non-finite logits")
            logits.append(row)
            token = int(row.argmax())
            generated_ids.append(token)
            if token in stop_ids:
                break
            if cache is None:
                raise RuntimeError("model returned no KV cache")
            current_ids = torch.tensor([[token]], device="cuda:0", dtype=torch.long)
            attention_mask = torch.cat((attention_mask, torch.ones_like(current_ids)), dim=1)

    directory = MOUNT / "runs" / run_id
    directory.mkdir(parents=True)
    filename = f"{case['id']}.npz"
    np.savez_compressed(
        directory / filename,
        input_ids=input_ids[0].numpy().astype(np.int64),
        prefill_positions=np.array([input_ids.shape[1] - 1], dtype=np.int64),
        prefill_logits=logits[0][None, :],
        step_logits=np.stack(logits),
        generated_ids=np.array(generated_ids, dtype=np.int64),
    )
    manifest = {
        "format": "muse-transformers-reference-v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "max_new_tokens": max_new_tokens,
        "cases": [{
            "id": case["id"],
            "messages": case["messages"],
            "file": filename,
            "input_length": input_ids.shape[1],
            "prefill_positions": [input_ids.shape[1] - 1],
            "generated_ids": generated_ids,
            "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
            "stopped_on_eos": generated_ids[-1] in stop_ids,
        }],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    VOLUME.commit()


@app.local_entrypoint()
def main(max_new_tokens: int, prompt: str = "") -> None:
    """Run a custom prompt or the saved smoke case and download the results."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if prompt:
        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")
        case = {"id": "custom_prompt", "messages": [{"role": "user", "content": prompt}]}
    else:
        case = json.loads((ROOT / "prompt.jsonl").read_text())
        if case.get("id") != "gsm8k_test_0001":
            raise ValueError("unexpected smoke case id")
    run_id = uuid.uuid4().hex
    capture.remote(case, max_new_tokens, run_id)
    destination = ROOT / "output" / run_id
    destination.mkdir(parents=True)
    for name in (f"{case['id']}.npz", "manifest.json"):
        with (destination / name).open("wb") as stream:
            for chunk in VOLUME.read_file(f"runs/{run_id}/{name}"):
                stream.write(chunk)
    print(destination)
