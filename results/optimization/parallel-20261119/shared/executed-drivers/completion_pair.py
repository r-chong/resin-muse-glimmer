"""Independent repeated validation of source variants on one physical H100."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
import modal

os.environ.setdefault('MODAL_IMAGE_BUILDER_VERSION','2025.06')


def execute(config):
    import importlib.metadata
    import random
    import subprocess
    import time
    from pathlib import Path
    started=time.monotonic()
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    assert hardware.startswith('NVIDIA H100 80GB HBM3,')
    result={'hardware':hardware,'variants':config['variants'],'qualification':{},'legs':[],'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
    out=Path('/experiment')

    def save():
        result['gpu_function_elapsed_s']=time.monotonic()-started
        (out/'result.json').write_text(json.dumps(result,indent=2))

    def run(variant,command,label):
        name=variant['name'];cache='/tmp/cache-'+name
        env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/tmp/hf','XDG_CACHE_HOME':cache,'TRITON_CACHE_DIR':cache+'/triton','TORCHINDUCTOR_CACHE_DIR':cache+'/inductor','PYTHONPATH':'/variants/'+name+':/variants/'+name+'/tests','RESIN_TEST_CHECKPOINT':config['checkpoint'],'RESIN_MUSE_STEPS':'16','RESIN_MUSE_REFERENCES':','.join(['/bench-traces/muse-reference/e0b3ae6b/gsm8k_test_0001.npz','/bench-traces/muse-reference/69d7fbf4/long_window.npz','/bench-traces/muse-reference/69d7fbf4/long_8k.npz']),'RESIN_MUSE_DUMP_DIR':'/experiment/diagnostics-'+name,**variant.get('environment',{})}
        (out/('diagnostics-'+name)).mkdir(exist_ok=True)
        lines=[];began=time.monotonic()
        with subprocess.Popen(command,cwd='/root',env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
            for line in process.stdout:lines.append(line);print(line,end='',flush=True)
            process.wait()
        (out/(label+'.log')).write_text(''.join(lines))
        return {'name':name,'command':command,'returncode':process.returncode,'elapsed_s':time.monotonic()-began,'output':''.join(lines)}

    try:
        for variant in config['variants']:
            name=variant['name']
            if variant.get('qualify'):
                command=['python','-m','pytest','-o','pythonpath=',*[f'/variants/{name}/{p}' for p in variant.get('extra_tests',[])],f'/variants/{name}/tests/test_muse_glimmer_qualified.py','-s','-v','--tb=short']
                result['qualification'][name]=run(variant,command,'qualify-'+name);save()
        randomizer=random.Random(41)
        for repeat in range(1,1 if config.get('completion_only') else 4):
            variants=list(config['variants']);randomizer.shuffle(variants)
            for variant in variants:
                name=variant['name']
                if result['qualification'].get(name,{}).get('returncode',0)!=0:continue
                label=f'round-{repeat}-{name}'
                path='/experiment/'+label+'.jsonl'
                print('PAIRED_LEG '+json.dumps({'repeat':repeat,'variant':name,'order':[v['name'] for v in variants]}),flush=True)
                command=['python','/root/scripts/bench.py','--backend','local','--model','muse-glimmer-30b','--checkpoint',config['checkpoint'],'--prompt-source','synthetic','--phases','prefill,decode','--prefill-context-lengths','4096,8192','--prefill-batch-sizes','1','--decode-context-lengths','8192','--decode-batch-sizes','1,2,4,8','--warmup-steps','5','--measure-steps','250','--repeats','1','--prefill-graph','disabled','--decode-graph','disabled','--no-overlap','--output',path]
                leg=run(variant,command,label);leg.update({'round':repeat,'order':[v['name'] for v in variants]})
                if Path(path).exists():leg['benchmark']=[json.loads(line) for line in Path(path).read_text().splitlines()]
                result['legs'].append(leg);save()
        for variant in config['variants']:
            if not variant.get('completion'):continue
            name=variant['name']
            if result['qualification'].get(name,{}).get('returncode',0)!=0:continue
            path=f'/experiment/completion-{name}.json'
            command=['python','/root/completion_bench.py','--backend','resin','--checkpoint',config['checkpoint'],'--output',path,'--repeats','3']
            leg=run(variant,command,'completion-'+name)
            if Path(path).exists():leg['completion']=json.loads(Path(path).read_text())
            result.setdefault('completions',[]).append(leg);save()
        return result
    finally:
        save();modal.Volume.from_name(config['volume']).commit()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--job-id',required=True);p.add_argument('--timeout',type=int,default=900);p.add_argument('--candidate',choices=['A2','B1','C']);p.add_argument('--completion-only',action='store_true');args=p.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    source=Path('/private/tmp/resin-opt-snapshots/baseline/resin-source');sys.path.insert(0,str(source))
    from deploy.modal_image import build_modal_image
    image=build_modal_image(source).add_local_file(source/'scripts/bench.py','/root/scripts/bench.py',copy=True).uv_pip_install('pytest>=8,<9').add_local_file(Path(__file__).with_name('completion_bench.py'),'/root/completion_bench.py',copy=True)
    variants=[{'name':'baseline','snapshot':'baseline','commit':'73532b9','qualify':True,'completion':True},{'name':'A2','snapshot':'a2','commit':'e1424dc','qualify':True},{'name':'B1','snapshot':'b1','commit':'3baaa0f','environment':{'RESIN_MUSE_DECODE_EXECUTION':'graphs'}},{'name':'C','snapshot':'c-final','commit':'66bb70a','qualify':True,'completion':True,'extra_tests':['tests/test_muse_glimmer_flashinfer.py']}]
    if args.candidate:
        variants=[v for v in variants if v['name'] in ['baseline',args.candidate]]
        for v in variants:
            v['completion']=False
            if v['name']=='baseline':v['qualify']=args.candidate=='C'
    if args.completion_only:
        variants=[v for v in variants if v['name'] in ['baseline','C']]
        for v in variants:v.update(qualify=v['name']=='C',completion=True)
    diagnostic=Path(__file__).with_name('test_executing_path.py').read_text().replace('from test_muse_glimmer_hf_parity import runner, references, _generate, _logit_summary','')
    diagnostic=diagnostic.replace('torch.from_numpy(expected).log_softmax(-1)','torch.from_numpy(expected).double().log_softmax(-1)').replace('torch.from_numpy(actual).log_softmax(-1)','torch.from_numpy(actual).double().log_softmax(-1)')
    diagnostic=diagnostic.replace("Path('/experiment/diagnostics').mkdir(exist_ok=True)","Path(os.environ['RESIN_MUSE_DUMP_DIR']).mkdir(exist_ok=True)").replace("'/experiment/diagnostics/executing-' + ref['name'] + '.npz'","str(Path(os.environ['RESIN_MUSE_DUMP_DIR']) / ('executing-' + ref['name'] + '.npz'))")
    for variant in variants:
        src=Path('/private/tmp/resin-opt-snapshots')/variant['snapshot']/'resin-source';name=variant['name']
        image=image.add_local_dir(src/'models/torch/resin_models_torch',f'/variants/{name}/resin_models_torch',copy=True,ignore=['**/__pycache__','**/*.pyc'])
        image=image.add_local_dir(src/'tests',f'/variants/{name}/tests',copy=True)
        combined=args.output/('qualified-'+name+'.py');combined.write_text((src/'tests/test_muse_glimmer_hf_parity.py').read_text()+'\n'+diagnostic)
        image=image.add_local_file(combined,f'/variants/{name}/tests/test_muse_glimmer_qualified.py',copy=True)
        variant['source_sha256']={str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((src/'models/torch/resin_models_torch').rglob('*.py'))}
    volume_name='resin-opt-'+args.job_id;volume=modal.Volume.from_name(volume_name,create_if_missing=True)
    config={'completion_only':args.completion_only,'variants':variants,'volume':volume_name,'checkpoint':'/models/resin-muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69'}
    (args.output/'request.json').write_text(json.dumps(config,indent=2))
    app=modal.App('resin-opt-'+args.job_id)
    worker=app.function(image=image,gpu='H100!',cpu=32,memory=(131072,262144),timeout=args.timeout,retries=0,max_containers=1,scaledown_window=2,single_use_containers=True,volumes={'/models/resin-muse-glimmer-30b':modal.Volume.from_name('resin-muse-glimmer-30b-models-v1').with_mount_options(read_only=True),'/bench-traces':modal.Volume.from_name('resin-benchmark-traces-v1').with_mount_options(read_only=True),'/experiment':volume})(execute)
    with modal.enable_output(),app.run():
        call=worker.spawn(config);(args.output/'function-call.json').write_text(json.dumps({'object_id':call.object_id}));result=call.get()
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'output':str(args.output),'elapsed_s':result['gpu_function_elapsed_s'],'legs':len(result['legs'])}))


if __name__=='__main__':main()
