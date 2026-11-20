"""Common token-ID completion benchmark for stock vLLM and Resin.

Both adapters time submit-to-final CPU-visible token events. These results are
separate from Resin's synchronized-step optimization microbenchmarks.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


class ResinAdapter:
    def __init__(self, checkpoint):
        from resin import Engine
        from resin_models_torch.muse_glimmer import MuseGlimmerTorchRunner
        self.runner = MuseGlimmerTorchRunner(checkpoint, revision=None, include_vision=True, prefill_chunk_size=8192)
        capture = getattr(self.runner, 'capture_cuda_graphs', None)
        if capture: capture()
        self.engine = Engine(self.runner, context_length=8451, max_running_requests=8, prefix_cache=False, enable_overlap=False)

    def cohort(self, prompts, steps):
        from resin import SamplingParams
        start = time.perf_counter()
        leases = [self.engine.submit(p, sampling_params=SamplingParams(max_output_tokens=steps, temperature=0.0, top_p=1.0, seed=41, stop_token_ids=[])) for p in prompts]
        mapping = {lease.request_id: i for i, lease in enumerate(leases)}
        arrivals = [[] for _ in prompts]
        tokens = [[] for _ in prompts]
        while self.engine.has_work():
            events = self.engine.step(allow_overlap=False)
            now = time.perf_counter() - start
            for event in events:
                index = mapping[event.request_id]
                tokens[index].append(event.token_id)
                arrivals[index].append(now)
        elapsed = time.perf_counter() - start
        assert [lease.take_result() for lease in leases] == tokens
        return tokens, arrivals, elapsed


class VllmAdapter:
    def __init__(self, checkpoint):
        from vllm import LLM
        self.llm = LLM(model=checkpoint, tokenizer=checkpoint, dtype='bfloat16', kv_cache_dtype='auto', tensor_parallel_size=1, max_model_len=8451, max_num_seqs=8, max_num_batched_tokens=8192, enable_chunked_prefill=True, enable_prefix_caching=False, seed=41, skip_tokenizer_init=True, disable_log_stats=True, gpu_memory_utilization=0.90, generation_config='vllm', limit_mm_per_prompt={'image': 0, 'video': 0})
        self.engine = self.llm.llm_engine
        self.counter = 0

    def cohort(self, prompts, steps):
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=steps, min_tokens=steps, ignore_eos=True, temperature=0.0, top_p=1.0, top_k=-1, seed=41, detokenize=False)
        arrivals = [[] for _ in prompts]
        tokens = [[] for _ in prompts]
        start = time.perf_counter()
        mapping = {}
        for i, prompt in enumerate(prompts):
            request = str(self.counter); self.counter += 1
            mapping[request] = i
            self.engine.add_request(request, {'prompt_token_ids': prompt}, params)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            now = time.perf_counter() - start
            for output in outputs:
                index = mapping[output.request_id]
                emitted = list(output.outputs[0].token_ids)
                added = len(emitted) - len(tokens[index])
                assert added >= 0
                arrivals[index].extend([now] * added)
                tokens[index] = emitted
        elapsed = time.perf_counter() - start
        return tokens, arrivals, elapsed


def main():
    import torch
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['resin', 'vllm'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    started = time.perf_counter()
    adapter = (ResinAdapter if args.backend == 'resin' else VllmAdapter)(args.checkpoint)
    torch.cuda.synchronize()
    result = {'backend': args.backend, 'startup_s': time.perf_counter()-started, 'timing': 'CPU-visible token events, submit all requests through final token; no HTTP/detokenization', 'prefill_chunk_tokens':8192, 'max_running_requests':8, 'context_capacity':8451, 'dtype':'bfloat16', 'prefix_cache':False, 'speculation':False, 'sampling':{'temperature':0,'top_p':1,'seed':41,'ignore_eos':True}, 'references':[], 'records':[], 'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}

    def save():
        args.output.write_text(json.dumps(result,indent=2))

    for reference in ['e0b3ae6b/gsm8k_test_0001', '69d7fbf4/long_window', '69d7fbf4/long_8k']:
        ref = np.load('/bench-traces/muse-reference/' + reference + '.npz')
        expected = ref['generated_ids'][:16].tolist()
        (tokens,), _, _ = adapter.cohort([ref['input_ids'].tolist()], len(expected))
        mismatch = next((i for i,(a,b) in enumerate(zip(tokens,expected)) if a!=b),None)
        result['references'].append({'reference':reference, 'matches':sum(a==b for a,b in zip(tokens,expected)), 'first_mismatch':mismatch, 'actual':tokens,'expected':expected})
        save()
        assert tokens == expected, 'stock engine token mismatch: ' + reference

    generator=torch.Generator().manual_seed(41)
    prompt=torch.randint(1000,30000,(8451,),generator=generator).tolist()
    result['prompt_sha256']=hashlib.sha256(np.array(prompt,dtype=np.int64).tobytes()).hexdigest()
    for phase,context,batch,steps in [('prefill',4096,1,1),('prefill',8192,1,1),*[('decode',8192,b,256) for b in [1,2,4,8]]]:
        prompts=[prompt[:context] for _ in range(batch)]
        adapter.cohort(prompts,steps)
        for repeat in range(args.repeats):
            tokens,arrivals,elapsed=adapter.cohort(prompts,steps)
            assert all(len(t)==steps for t in tokens)
            r={'phase':phase,'context_length':context,'batch_size':batch,'repeat':repeat+1,'output_tokens':batch*steps,'elapsed_s':elapsed,'completion_output_tokens_s_gpu':batch*steps/elapsed,'ttft_s':[a[0] for a in arrivals],'tokens':tokens,'arrivals_s':arrivals,'gpu_count':1}
            if steps>1:
                r['per_request_tpot_s']=[(a[-1]-a[0])/(steps-1) for a in arrivals]
                r['completion_intervals_s']=[[y-x for x,y in zip(a,a[1:])] for a in arrivals]
                r['post_warmup_window_s']=max(a[-1] for a in arrivals)-min(a[5] for a in arrivals)
                r['post_warmup_output_tokens_s_gpu']=batch*250/r['post_warmup_window_s']
            result['records'].append(r);save()
            print('COMPLETION_BENCH_RESULT '+json.dumps({k:v for k,v in r.items() if k not in ['tokens','arrivals_s','completion_intervals_s']}),flush=True)
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
    result['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
    save()


if __name__=='__main__': main()
