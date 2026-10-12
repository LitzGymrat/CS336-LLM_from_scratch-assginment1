"""从真实 SQLite CUDA launch 关联生成 Nsight 分析表；不按 GPU 时间重叠猜标签。"""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

import pandas as pd
from cs336_systems.reporting import save_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/nsight_run/analysis'
PLAN=ROOT/'outputs/nsight_plan'


def operation(name):
    if re.search(r'gemm|gemv|cublas|cutlass',name,re.I):return 'matmul'
    if 'Reduce' in name or 'reduce' in name:return 'reduction'
    if 'masked_fill' in name or 'triu' in name:return 'mask'
    if 'index' in name.lower() or 'gather' in name.lower() or 'scatter' in name.lower():return 'index'
    if 'elementwise' in name:return 'elementwise'
    return 'other'


def load_trace(row):
    prefix=row['profile_prefix'];c=sqlite3.connect(ROOT/(prefix+'.sqlite'))
    names=dict(c.execute('SELECT id,value FROM StringIds'))
    ranges={}
    for rid,start,end,text,text_id,tid in c.execute('SELECT rowid,start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE eventType=59 ORDER BY start'):
        label=text or names[text_id]
        ranges.setdefault(label,[]).append({'id':rid,'start':start,'end':end,'tid':tid})
    starts={label:[r['start'] for r in rs] for label,rs in ranges.items()}
    def associations(start,end,tid):
        matched={}
        for label,rs in ranges.items():
            i=bisect_right(starts[label],start)-1
            # backward 的 CUDA launch 由 autograd worker 发起。该进程串行执行
            # 一个 step，按 CPU 阶段内的 launch 时间关联，不按 GPU 时间重叠关联。
            if i>=0 and end<=rs[i]['end']:
                matched[label]=i
        assert 'measurement' in matched and 'step' in matched,(start,end,matched)
        return matched
    events=[]
    # 本次 nsys trace 的 launch 全部记录在 CUDA runtime 表，逐事件验证一一关联。
    for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','memcpy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
        columns='k.demangledName' if kind=='kernel' else 'NULL'
        sql=f'''SELECT k.rowid,k.start,k.end,k.correlationId,{columns},r.start,r.end,r.globalTid
        FROM {table} k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
        ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)'''
        records=c.execute(sql).fetchall()
        assert len(records)==c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
        assert len({x[0] for x in records})==len(records)
        for rid,start,end,correlation,name_id,api_start,api_end,tid in records:
            matched=associations(api_start,api_end,tid)
            assert sum(label in matched for label in ['forward','loss','backward','optimizer'])<=1
            events.append(dict(event_id=rid,kind=kind,start_ns=start,end_ns=end,duration_ns=end-start,
                kernel_name=names[name_id] if name_id is not None else '',correlation_id=correlation,
                api_start_ns=api_start,api_end_ns=api_end,api_global_tid=tid,
                launch_thread_matches_step_range=tid==ranges['step'][matched['step']]['tid'],**matched))
    c.close()
    return events,ranges


def analyze(row):
    events,ranges=load_trace(row);n=int(row['steps']);name=Path(row['profile_prefix']).name
    meta={k:row[k] for k in ['config_id','model','seq_len','mode']}
    df=pd.DataFrame(events)
    df.to_csv(OUT/f'{name}_events.csv',index=False)
    kernels=[e for e in events if e['kind']=='kernel']
    stage_rows=[];kernel_rows=[];count_checks=[]
    for scope,label in [('total','step'),('forward','forward'),('loss','loss'),('backward','backward'),('optimizer','optimizer')]:
        if label not in ranges:continue
        selected=[e for e in events if label in e]
        ks=[e for e in selected if e['kind']=='kernel']
        sums=[];spans=[];matmuls=[]
        for i in range(n):
            per=[e for e in selected if e[label]==i]
            spans.append((max(e['end_ns'] for e in per)-min(e['start_ns'] for e in per))/1e6)
            sums.append(sum(e['duration_ns'] for e in per if e['kind']=='kernel')/1e6)
            matmuls.append(sum(e['duration_ns'] for e in per if e['kind']=='kernel' and operation(e['kernel_name'])=='matmul')/1e6)
        total=sum(sums);matmul=sum(matmuls)
        stage_rows.append(dict(**meta,scope=scope,steps=n,gpu_span_mean_ms=statistics.mean(spans),
            gpu_span_std_ms=statistics.stdev(spans),gpu_kernel_sum_mean_ms=statistics.mean(sums),
            matmul_gpu_mean_ms=statistics.mean(matmuls),other_kernel_gpu_mean_ms=(total-matmul)/n,
            matmul_percent=100*matmul/total,other_kernel_percent=100*(total-matmul)/total,source_report=row['profile_prefix']+'.nsys-rep'))
        groups={}
        for e in ks:groups.setdefault(e['kernel_name'],[]).append(e)
        for kernel,es in sorted(groups.items(),key=lambda x:-sum(e['duration_ns'] for e in x[1])):
            counts=[sum(e['step']==i for e in es) for i in range(n)]
            assert len(set(counts))==1,(name,scope,kernel,counts)
            count_checks.append({'scope':scope,'kernel_name':kernel,'calls_by_step':counts})
            softmax=sum(e['duration_ns'] for e in es if 'attention_softmax' in e)
            attention=sum(e['duration_ns'] for e in es if 'attention' in e)
            duration=sum(e['duration_ns'] for e in es)
            evidence=f'CUDA runtime correlationId and same process, launch time within serial CPU stage; attention {attention/duration:.1%}, attention_softmax {softmax/duration:.1%} of this kernel time'
            kernel_rows.append(dict(**meta,scope=scope,kernel_name=kernel,operation_class=operation(kernel),
                calls_total=len(es),calls_per_step=len(es)/n,gpu_total_ms=duration/1e6,
                gpu_avg_us=duration/len(es)/1e3,gpu_max_us=max(e['duration_ns'] for e in es)/1e3,
                percent_scope_kernel_time=100*duration/1e6/total,
                attention_gpu_ms=attention/1e6,attention_softmax_gpu_ms=softmax/1e6,
                source_report=row['profile_prefix']+'.nsys-rep',attribution_evidence=evidence))
    # 同一原始 kernel 只算一次；与官方 kernel summary 逐名称、次数、累计时间全量核对。
    official=pd.read_csv(OUT/f'{name}_cuda_gpu_kern_sum.csv')
    own={r['kernel_name']:r for r in kernel_rows if r['scope']=='total'}
    assert set(official['Name'])==set(own)
    for r in official.to_dict('records'):
        observed=own[r['Name']]
        assert observed['calls_total']==r['Instances']
        assert abs(observed['gpu_total_ms']*1e6-r['Total Time (ns)'])<0.01
    projected=pd.read_csv(OUT/f'{name}_nvtx_gpu_proj_sum.csv')
    projection_checks=[]
    for r in stage_rows:
        label='step' if r['scope']=='total' else r['scope']
        per=[e for e in events if label in e]
        worker_ops=sum(not e['launch_thread_matches_step_range'] for e in per)
        official_row=projected.loc[projected['Range']==':'+label,'Proj Avg (ns)']
        official_span=float(official_row.iloc[0])/1e6 if len(official_row) else None
        if worker_ops==0:
            assert official_span is not None
            assert abs(r['gpu_span_mean_ms']-official_span)<1e-6,(name,label,r['gpu_span_mean_ms'],official_span)
        projection_checks.append(dict(scope=r['scope'],worker_thread_gpu_ops=worker_ops,
            own_all_threads_span_ms=r['gpu_span_mean_ms'],official_same_thread_span_ms=official_span))
    att=None
    if row['mode']=='forward':
        s=int(row['seq_len']);b=int(row['batch_size']);d=int(row['d_model']);h=int(row['num_heads']);l=int(row['num_layers'])
        soft_time=sum(e['duration_ns'] for e in kernels if 'attention_softmax' in e)/n/1e6
        mat_time=sum(e['duration_ns'] for e in kernels if 'attention' in e and operation(e['kernel_name'])=='matmul')/n/1e6
        # max 比较也计作一次标量操作，exp/div 各计一次；这是近似运算计数，不是硬件指令数。
        soft_ops=l*b*h*s*(5*s-2);mat_flops=l*(8*b*s*d*d+4*b*s*s*d)
        att=dict(**{k:row[k] for k in ['config_id','model','seq_len','batch_size','num_layers','d_model','num_heads']},
            scope='all MSA layers within one forward',softmax_gpu_mean_ms=soft_time,msa_matmul_gpu_mean_ms=mat_time,
            softmax_flops=soft_ops,msa_matmul_flops=mat_flops,
            flops_counting_convention='softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2',
            time_ratio_softmax_to_matmul=soft_time/mat_time,flops_ratio_softmax_to_matmul=soft_ops/mat_flops,
            source_report=row['profile_prefix']+'.nsys-rep')
    validation=dict(source_report=row['profile_prefix']+'.nsys-rep',all_kernel_names_counts_times_match_official_summary=True,
        same_thread_stage_gpu_spans_match_official_projection=True,
        all_gpu_ops_have_launch_inside_exactly_one_serial_step=True,
        all_kernel_launches_joined_once=True,projection_checks=projection_checks,
        all_per_step_kernel_call_counts_constant=True,kernel_event_count=len(kernels),count_checks=count_checks)
    (OUT/f'{name}_analysis_validation.json').write_text(json.dumps(validation,indent=2))
    print('ANALYZED',name,'kernels',len(kernels),'validated against official summaries')
    return stage_rows,kernel_rows,att


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    rows=pd.read_csv(PLAN/'experiment_manifest.csv',keep_default_na=False).to_dict('records')
    if args.pilot:
        rows=[next(r for r in rows if r['model']=='small' and int(r['seq_len'])==512 and r['mode']=='forward')]
    else:rows=[r for r in rows if r['status']=='completed']
    stages=[];kernels=[];attention=[]
    for row in rows:
        ss,ks,att=analyze(row);stages.extend(ss);kernels.extend(ks)
        if att:attention.append(att)
    target=OUT/'pilot' if args.pilot else PLAN
    save_table(stages,target/'stage_observations');save_table(kernels,target/'kernel_observations');save_table(attention,target/'attention_observations')
    print('Tables exported:',target)


if __name__=='__main__':main()
