"""执行已经选定的 Nsight 实验：串行子进程、逐项日志和 CSV 状态。"""
import csv
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'outputs/nsight_plan'
OUT = ROOT / 'outputs/nsight_run'
MODELS = {'small': (768,3072,12,12), 'medium': (1024,4096,24,16)}
MODES = ['forward','forward_backward','train']


def read_csv(name):
    with (PLAN/name).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (PLAN/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def command(model,mode,length,warmup,steps,prefix,nvtx=False):
    d,f,l,h=MODELS[model]
    args=['uv','run','python','measure_gpu_memory.py','--device','cuda','--mode',mode,
          '--vocab_size','10000','--d_model',str(d),'--d_ff',str(f),'--num_layers',str(l),
          '--num_heads',str(h),'--batch_size','4','--context_len',str(length),
          '--warmup',str(warmup),'--steps',str(steps),'--output_prefix',prefix]
    if nvtx: args.append('--nvtx')
    return args


def execute(args, log_path):
    print(time.strftime('%H:%M:%S'), 'START', shlex.join(args),flush=True)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['NSYS_NVTX_PROFILER_REGISTER_ONLY']='0'
    with (ROOT/log_path).open('w') as log:
        job=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try:
                code=job.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                print(time.strftime('%H:%M:%S'),'RUNNING',log_path,flush=True)
    content=(ROOT/log_path).read_text()
    status='completed' if code==0 else 'oom' if 'CUDA out of memory' in content or 'torch.OutOfMemoryError' in content else 'error'
    peaks={key: re.search(label+r'=([0-9.]+)',content).group(1) if re.search(label+r'=([0-9.]+)',content) else ''
           for key,label in [('peak_allocated_gib','peak_allocated_GiB'),('peak_reserved_gib','peak_reserved_GiB')]}
    print(time.strftime('%H:%M:%S'),'END',status,'exit',code,peaks,flush=True)
    if status=='error':
        print(content[-3000:],flush=True)
        raise RuntimeError(f'Non-OOM error: {log_path}')
    return {'status':status,'exit_code':code,**peaks}


def export_and_validate(row):
    prefix=row['profile_prefix']
    name=Path(prefix).name
    export=['nsys','stats','--report','cuda_gpu_kern_sum,nvtx_gpu_proj_sum','--format','csv',
            '--output',f'outputs/nsight_run/analysis/{name}',prefix+'.nsys-rep']
    target=ROOT/f'outputs/nsight_run/analysis/{name}_nvtx_gpu_proj_sum.csv'
    if not target.exists(): execute(export,f'outputs/nsight_run/logs/{name}_stats.log')
    db=sqlite3.connect(ROOT/(prefix+'.sqlite'))
    counts=dict(db.execute('''SELECT COALESCE(n.text,s.value),COUNT(*)
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE eventType=59 GROUP BY COALESCE(n.text,s.value)'''))
    expected={'measurement':1,'step':10,'forward':10,'attention':10*int(row['num_layers']),
              'attention_softmax':10*int(row['num_layers'])}
    if row['mode']!='forward': expected.update(loss=10,backward=10)
    if row['mode']=='train': expected['optimizer']=10
    assert {label:counts.get(label,0) for label in expected}==expected,(name,counts,expected)
    kernels=db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    assert kernels>0
    validation={'profile':prefix+'.nsys-rep','nvtx_counts':counts,'expected_counts':expected,
                'kernel_events':kernels,'formal_steps':10,'valid':True}
    (OUT/'analysis'/f'{name}_validation.json').write_text(json.dumps(validation,indent=2))
    db.close()
    print('VALIDATED',name,counts,'kernel events',kernels,flush=True)


def main():
    probes=read_csv('capacity_probe_manifest.csv')
    experiments=read_csv('experiment_manifest.csv')
    # 第一份 trace 已人工启动并核对；直接接续，不覆盖或重复采集。
    first=next(r for r in experiments if (r['model'],r['seq_len'],r['mode'])==('small','512','forward'))
    content=(ROOT/first['stdout_log']).read_text()
    first.update(status='completed',exit_code=0,
                 peak_allocated_gib=re.search(r'peak_allocated_GiB=([0-9.]+)',content).group(1),
                 peak_reserved_gib=re.search(r'peak_reserved_GiB=([0-9.]+)',content).group(1))
    export_and_validate(first);write_csv('experiment_manifest.csv',experiments)
    if '--resume-profiles' in sys.argv:
        capacities=read_csv('capacity_results.csv')
    else:
        capacities=[]
        for model in MODELS:
            for mode in MODES:
                length=1024;last=512;success_log=f'outputs/benchmark/{model}_{mode}.csv';a='';r=''
                while True:
                    row=next((x for x in probes if (x['model'],x['mode'],int(x['seq_len']))==(model,mode,length)),None)
                    if row is None:
                        row={key:'' for key in probes[0]};row.update(model=model,mode=mode,seq_len=length,warmup=1,steps=2,
                            prior_success_seq_len=last,prior_success_csv=success_log,status='not_run',
                            stdout_log=f'outputs/nsight_run/logs/probe_{model}_s{length}_{mode}.log')
                        row['command']=shlex.join(command(model,mode,length,1,2,f'outputs/nsight_run/memory_probes/{model}_s{length}_{mode}'))
                        probes.append(row)
                    row['status']='running';write_csv('capacity_probe_manifest.csv',probes)
                    result=execute(shlex.split(row['command']),row['stdout_log']);row.update(result)
                    write_csv('capacity_probe_manifest.csv',probes)
                    if result['status']=='oom':
                        capacities.append(dict(model=model,mode=mode,precision='torch.float32',batch_size=4,
                            longest_success_seq_len=last,first_oom_seq_len=length,success_peak_allocated_gib=a,
                            success_peak_reserved_gib=r,success_log=success_log,oom_log=row['stdout_log'],
                            profile_longest_success_seq_len=''))
                        break
                    last=length;success_log=row['stdout_log'];a=result['peak_allocated_gib'];r=result['peak_reserved_gib'];length*=2
                for x in probes:
                    if x['model']==model and x['mode']==mode and x['status']=='not_run':x['status']='not_needed_after_oom'
                write_csv('capacity_probe_manifest.csv',probes)
                write_csv('capacity_results.csv',capacities)
    for cap in capacities:
        for row in experiments:
            if row['model']==cap['model'] and row['mode']==cap['mode'] and int(row['seq_len'])>int(cap['longest_success_seq_len']):
                row['status']='capacity_oom'
        longest=int(cap['longest_success_seq_len'])
        if longest>1024:
            row=next(x for x in experiments if x['model']==cap['model'] and x['mode']==cap['mode']).copy()
            prefix=f"outputs/nsight_run/profiles/{cap['model']}_s{longest}_{cap['mode']}"
            row.update(config_id=f"{cap['model']}_max",seq_len=longest,status='not_run',profile_prefix=prefix,
                stdout_log=f"outputs/nsight_run/logs/{Path(prefix).name}.log",exit_code='',peak_allocated_gib='',peak_reserved_gib='',
                prior_python_forward_csv='',prior_python_forward_mean_ms='',prior_python_forward_std_ms='')
            row['capture_command']='' # 构造出的真实命令将在启动前填入。
            experiments.append(row)
    order={'small':0,'medium':1}
    experiments.sort(key=lambda x:(order[x['model']],{512:0,256:1}.get(int(x['seq_len']),2),int(x['seq_len']),MODES.index(x['mode'])))
    write_csv('experiment_manifest.csv',experiments)
    for row in experiments:
        if row['status']=='not_run':
            args=['uv','run','nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                '--capture-range=nvtx','--nvtx-capture=measurement','--capture-range-end=stop',
                '--output='+row['profile_prefix'],'--']+command(row['model'],row['mode'],int(row['seq_len']),5,10,row['profile_prefix']+'_timing',True)[2:]
            row['capture_command']='NSYS_NVTX_PROFILER_REGISTER_ONLY=0 '+shlex.join(args)
            row['status']='running';write_csv('experiment_manifest.csv',experiments)
            row.update(execute(args,row['stdout_log']))
            write_csv('experiment_manifest.csv',experiments)
            if row['status']=='completed':export_and_validate(row)
    for cap in capacities:
        cap['profile_longest_success_seq_len']=max(int(r['seq_len']) for r in experiments
            if r['model']==cap['model'] and r['mode']==cap['mode'] and r['status']=='completed')
    write_csv('capacity_results.csv',capacities)
    (OUT/'execution_complete.json').write_text(json.dumps({'complete':True,'profiles':sum(r['status']=='completed' for r in experiments),
         'capacity_oom':sum(r['status']=='capacity_oom' for r in experiments),'profiler_oom':sum(r['status']=='oom' for r in experiments)},indent=2))
    print('ALL EXPERIMENTS FINISHED',flush=True)


if __name__=='__main__':main()
