"""Unmodified released vLLM image, bounded Modal execution."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
import modal

os.environ.setdefault('MODAL_IMAGE_BUILDER_VERSION','2025.06')


def execute(config):
    import importlib.metadata
    import subprocess
    import time
    from pathlib import Path
    began=time.monotonic()
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip() if not config['prepare'] else None
    if config['prepare']:
        return {'version':importlib.metadata.version('vllm'),'mode':'CPU image preparation'}
    assert hardware.startswith('NVIDIA H100 80GB HBM3,')
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/tmp/hf','XDG_CACHE_HOME':'/tmp/cache','TORCHINDUCTOR_CACHE_DIR':'/tmp/inductor','TRITON_CACHE_DIR':'/tmp/triton','VLLM_NO_USAGE_STATS':'1'}
    command=['python','/root/completion_bench.py','--backend','vllm','--checkpoint','/models/resin-muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69','--output','/experiment/completion.json','--repeats','3']
    output=[]
    try:
        with subprocess.Popen(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
            for line in process.stdout: output.append(line);print(line,end='',flush=True)
            process.wait()
        result={'hardware':hardware,'version':importlib.metadata.version('vllm'),'returncode':process.returncode,'gpu_function_elapsed_s':time.monotonic()-began,'output':''.join(output),'command':command}
        if Path('/experiment/completion.json').exists(): result['completion']=json.loads(Path('/experiment/completion.json').read_text())
        Path('/experiment/result.json').write_text(json.dumps(result,indent=2))
        return result
    finally:
        modal.Volume.from_name(config['volume']).commit()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--job-id',required=True);args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    image=(modal.Image.from_registry('nvidia/cuda:13.3.1-devel-ubuntu22.04',add_python='3.12')
           .entrypoint([]).uv_pip_install('vllm==0.34.0')
           .add_local_file(Path(__file__).with_name('completion_bench.py'),'/root/completion_bench.py',copy=True))
    volume_name='resin-opt-'+args.job_id
    volume=modal.Volume.from_name(volume_name,create_if_missing=True)
    app=modal.App('resin-opt-'+args.job_id)
    worker=app.function(image=image,gpu=None if args.prepare else 'H100!',cpu=1 if args.prepare else 32,memory=1024 if args.prepare else 262144,timeout=900,retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,volumes={'/models/resin-muse-glimmer-30b':modal.Volume.from_name('resin-muse-glimmer-30b-models-v1').with_mount_options(read_only=True),'/bench-traces':modal.Volume.from_name('resin-benchmark-traces-v1').with_mount_options(read_only=True),'/experiment':volume})(execute)
    with modal.enable_output(),app.run():
        call=worker.spawn({'prepare':args.prepare,'volume':volume_name});(args.output/'function-call.json').write_text(json.dumps({'object_id':call.object_id}));result=call.get()
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['output','completion']}))


if __name__=='__main__':main()
