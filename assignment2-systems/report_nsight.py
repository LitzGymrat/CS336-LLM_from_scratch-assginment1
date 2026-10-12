"""从分析 CSV 生成中文报告和可复用的表/图。"""
from pathlib import Path
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cs336_systems.reporting import save_table

ROOT=Path(__file__).resolve().parent
PLAN=ROOT/'outputs/nsight_plan'
OUT=ROOT/'outputs/nsight_run/analysis'
DOC=ROOT/'docs/nsight_report.md'


def table(df):
    def cell(x):
        if pd.isna(x):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.3f}'
        return str(x).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
        '| '+' | '.join(['---']*len(df.columns))+' |']+
        ['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])


def savefig(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def short(name):
    if 'gemm' in name.lower():return name
    vector='vectorized' if 'vectorized_elementwise' in name else 'elementwise' if 'elementwise' in name else ''
    for tag in ['MulFunctor','DivFunctor','SubFunctor','CUDAFunctor_add','exp_kernel_cuda','sigmoid_kernel_cuda',
                'masked_fill_kernel','direct_copy_kernel_cuda','Reduce','index','triu','sqrt','pow','radix','sort']:
        if tag.lower() in name.lower():
            variant='scalar × tensor' if 'BUnaryFunctor' in name else 'tensor × tensor' if 'BinaryFunctor' in name and tag=='MulFunctor' else ''
            return ' / '.join(x for x in [vector,tag,variant] if x)
    return name[:90]


def main():
    manifest=pd.read_csv(PLAN/'experiment_manifest.csv');completed=manifest[manifest.status=='completed']
    assert len(completed)==15 and (manifest.status=='capacity_oom').sum()==3
    stages=pd.read_csv(PLAN/'stage_observations.csv');kernels=pd.read_csv(PLAN/'kernel_observations.csv');attention=pd.read_csv(PLAN/'attention_observations.csv')
    capacities=pd.read_csv(PLAN/'capacity_results.csv')
    configs=sorted({(r.model,int(r.seq_len)) for r in completed.itertuples()},key=lambda x:({'small':0,'medium':1}[x[0]],x[1]))
    def stage(model,s,mode,scope):
        return stages[(stages.model==model)&(stages.seq_len==s)&(stages['mode']==mode)&(stages.scope==scope)].iloc[0]
    def top(model,s,mode):
        return kernels[(kernels.model==model)&(kernels.seq_len==s)&(kernels['mode']==mode)&(kernels.scope=='total')].sort_values('gpu_total_ms',ascending=False).iloc[0]
    cap_rows=[]
    for r in capacities.to_dict('records'):
        p=completed[(completed.model==r['model'])&(completed['mode']==r['mode'])&(completed.seq_len==r['profile_longest_success_seq_len'])].iloc[0]
        cap_rows.append({'model':r['model'],'mode':r['mode'],'last_success_S':r['longest_success_seq_len'],
            'first_OOM_S':r['first_oom_seq_len'],'profile_peak_allocated_GiB':p.peak_allocated_gib,
            'profile_peak_reserved_GiB':p.peak_reserved_gib})
    cap_table=save_table(cap_rows,OUT/'capacity_table')
    comparison=[];hot=[];nonmat=[];answers=[]
    for model,s in configs:
        f=stage(model,s,'forward','forward');t=stage(model,s,'train','total');best=top(model,s,'forward');fb=top(model,s,'forward_backward')
        timing=pd.read_csv(ROOT/f'outputs/nsight_run/profiles/{model}_s{s}_forward_timing.csv').iloc[0]
        old_path=ROOT/f'outputs/benchmark/{model}_forward.csv'
        old=pd.read_csv(old_path).iloc[0] if s==512 else None
        comparison.append({'model':model,'seq_len':s,'prior_python_mean_ms':old.mean_ms if old is not None else None,
            'prior_python_std_ms':old.std_ms if old is not None else None,'nsight_gpu_span_mean_ms':f.gpu_span_mean_ms,
            'nsight_gpu_span_std_ms':f.gpu_span_std_ms,'kernel_sum_mean_ms':f.gpu_kernel_sum_mean_ms,
            'profiled_python_mean_ms':timing.mean_ms,'profiled_python_std_ms':timing.std_ms})
        hot.append({'model':model,'seq_len':s,'forward_top_kernel':best.kernel_name,
            'forward_total_ms_10_steps':best.gpu_total_ms,'forward_calls_per_step':best.calls_per_step,
            'forward_kernel_percent':best.percent_scope_kernel_time,'forward_backward_top_kernel':fb.kernel_name,
            'forward_backward_total_ms_10_steps':fb.gpu_total_ms,'forward_backward_calls_per_step':fb.calls_per_step,
            'forward_backward_kernel_percent':fb.percent_scope_kernel_time,'same_top_kernel':best.kernel_name==fb.kernel_name})
        others=kernels[(kernels.model==model)&(kernels.seq_len==s)&(kernels['mode']=='forward')&(kernels.scope=='total')&(kernels.operation_class!='matmul')].sort_values('gpu_total_ms',ascending=False).head(5)
        for r in others.to_dict('records'):
            association='softmax 内' if abs(r['attention_softmax_gpu_ms']-r['gpu_total_ms'])<1e-8 else 'MSA 内、softmax 外' if abs(r['attention_gpu_ms']-r['gpu_total_ms'])<1e-8 else 'MSA 外' if r['attention_gpu_ms']==0 else '多个区域共用'
            nonmat.append({'model':model,'seq_len':s,'kernel':short(r['kernel_name']),'gpu_ms_per_forward':r['gpu_total_ms']/10,
                'percent_forward_kernel_time':r['percent_scope_kernel_time'],'calls_per_forward':r['calls_per_step'],
                'observed_range':association,'full_kernel_name':r['kernel_name']})
        att=attention[(attention.model==model)&(attention.seq_len==s)].iloc[0]
        old_clause=f'旧 Python 测得 {old.mean_ms:.2f} ms，两者相差 {(f.gpu_span_mean_ms/old.mean_ms-1)*100:+.1f}%。' if old is not None else '该长度没有同配置的旧 Python 结果。'
        answers.append({'model':model,'seq_len':s,
            'a':f'前向 GPU 执行跨度为 {f.gpu_span_mean_ms:.2f} ms，kernel 累计时间为 {f.gpu_kernel_sum_mean_ms:.2f} ms。'+old_clause,
            'b':f'前向累计时间最多的是 {best.kernel_name}，每次前向调用 {best.calls_per_step:g} 次，占 kernel 时间 {best.percent_scope_kernel_time:.1f}%。前向加反向的榜首为 {fb.kernel_name}'+('，名称相同。' if best.kernel_name==fb.kernel_name else '，与前向不同。'),
            'c':f'主要非矩阵乘法 kernel 包括 '+ '、'.join(f"{short(r.kernel_name)}（{r.percent_scope_kernel_time:.1f}%）" for r in others.head(3).itertuples())+'；具体标签归属见 (c) 表。',
            'd':f'完整训练的矩阵乘法占比为 {t.matmul_percent:.1f}%，前向为 {f.matmul_percent:.1f}%，变化 {t.matmul_percent-f.matmul_percent:+.1f} 个百分点。优化器阶段没有矩阵乘法，反向也增加了逐元素与归约等 kernel。',
            'e':f'全部 MSA 层的 softmax 与矩阵乘法分别耗时 {att.softmax_gpu_mean_ms:.3f}、{att.msa_matmul_gpu_mean_ms:.3f} ms，时间比为 {att.time_ratio_softmax_to_matmul:.2%}。按本文标量运算约定，运算量比仅 {att.flops_ratio_softmax_to_matmul:.2%}。'})
    comparison=save_table(comparison,OUT/'forward_comparison');hot=save_table(hot,OUT/'top_kernels');nonmat=save_table(nonmat,OUT/'nonmatmul_kernels');answers=save_table(answers,OUT/'answers_per_config')
    hot_display=hot[['model','seq_len','forward_top_kernel','forward_calls_per_step','forward_kernel_percent','forward_backward_top_kernel','forward_backward_kernel_percent']].copy()
    for column in ['forward_top_kernel','forward_backward_top_kernel']:
        hot_display[column]=hot_display[column].map(short)
    # 旧 Python、GPU span、kernel sum 分开显示，误差线只用于 span/Python 的标准差。
    paired=comparison[comparison.seq_len==512];x=np.arange(len(paired));fig,ax=plt.subplots(figsize=(8,4.5))
    for i,(value,error,label) in enumerate([('prior_python_mean_ms','prior_python_std_ms','Prior Python wall'),('nsight_gpu_span_mean_ms','nsight_gpu_span_std_ms','Nsight GPU span'),('kernel_sum_mean_ms',None,'Kernel sum')]):
        ax.bar(x+(i-1)*.24,paired[value],.24,yerr=paired[error] if error else None,label=label,capsize=4)
    ax.set_xticks(x,paired.model);ax.set(ylabel='ms / forward',title='RTX 4090 | FP32 | B=4 | S=512 | grad enabled');ax.legend();ax.grid(axis='y',alpha=.2);savefig(fig,'forward_comparison')
    # 每种模型各一张图，前向独立采集与训练阶段都用各自 kernel 时间作分母。
    for model in ['small','medium']:
        lengths=[s for m,s in configs if m==model];fig,axes=plt.subplots(1,len(lengths),figsize=(5*len(lengths),5),sharex=True,sharey=True)
        axes=np.atleast_1d(axes)
        labels=['Forward only','Train: forward','Train: loss','Train: backward','Train: AdamW','Train: total']
        for ax,s in zip(axes,lengths):
            values=[stage(model,s,'forward','forward').matmul_percent]+[stage(model,s,'train',scope).matmul_percent for scope in ['forward','loss','backward','optimizer','total']]
            y=np.arange(len(values));ax.barh(y,values,color='#397aab',label='Matmul');ax.barh(y,100-np.array(values),left=values,color='#e6a24b',label='Other kernels')
            for j,v in enumerate(values):ax.text(min(v+1,94),j,f'{v:.1f}%',va='center',fontsize=9)
            ax.set_yticks(y,labels);ax.set_xlim(0,100);ax.set_title(f'{model} | S={s}');ax.set_xlabel('% of kernel GPU time');ax.grid(axis='x',alpha=.2)
        axes[0].invert_yaxis();axes[-1].legend(loc='upper left',bbox_to_anchor=(0,-.16));fig.suptitle('RTX 4090 | FP32 | B=4 | 10 measured steps');savefig(fig,f'{model}_matmul_share')
    attention=attention.sort_values(['model','seq_len'],key=lambda c:c.map({'small':0,'medium':1}) if c.name=='model' else c)
    labels=[f'{r.model}\nS={r.seq_len}' for r in attention.itertuples()];x=np.arange(len(attention));fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    axes[0].bar(x,attention.softmax_gpu_mean_ms,.4,label='Softmax',color='#e6a24b');axes[0].bar(x+.4,attention.msa_matmul_gpu_mean_ms,.4,label='MSA matmul',color='#397aab');axes[0].set_xticks(x+.2,labels);axes[0].set(ylabel='ms / forward (all MSA layers)',title='Measured GPU kernel time');axes[0].legend()
    axes[1].bar(x-.18,attention.time_ratio_softmax_to_matmul*100,.36,label='Time ratio');axes[1].bar(x+.18,attention.flops_ratio_softmax_to_matmul*100,.36,label='Operation-count ratio');axes[1].set_xticks(x,labels);axes[1].set(ylabel='softmax / MSA matmul (%)',title='Time vs operation count');axes[1].legend()
    axes[1].set_yscale('log');axes[1].set_ylim(.1,300)
    axes[1].set_yticks([.1,1,10,100],['0.1','1','10','100'])
    for ax in axes:ax.grid(axis='y',alpha=.2)
    savefig(fig,'attention_comparison')
    paths='\n'.join(f"- [{Path(r.profile_prefix).name}.nsys-rep](../{r.profile_prefix}.nsys-rep)" for r in completed.itertuples())
    at=attention[['model','seq_len','softmax_gpu_mean_ms','msa_matmul_gpu_mean_ms','time_ratio_softmax_to_matmul','flops_ratio_softmax_to_matmul']].copy()
    at['time_ratio_softmax_to_matmul']*=100;at['flops_ratio_softmax_to_matmul']*=100
    at=at.rename(columns={'time_ratio_softmax_to_matmul':'time_ratio_%','flops_ratio_softmax_to_matmul':'operation_ratio_%'})
    nonmat_display=nonmat.drop(columns=['full_kernel_name'])
    report=[
        '# Nsight Systems 实验报告\n',
        '2026-10-08；RTX 4090 24 GiB。完成 **15 份正式 trace**、9 次容量探测；small 的三档长度为 256/512/1024，medium 的两档为 256/512。medium/1024 三种模式均实际 OOM；按本次选择保持 batch=4，因此 **medium 尚未满足作业要求的三档成功长度**，不把 OOM 当作第三档测量。\n',
        '## 实验条件与时间口径\n',
        '模型使用你自己的 A1 实现。small=(d=768,f=3072,L=12,H=12)，medium=(1024,4096,24,16)；V=10,000，B=4，FP32，pre-norm、SwiGLU、RoPE。warmup=5、正式 steps=10，逐配置独立进程、串行运行；梯度开启，前向模式与旧实验保持一致，因此其显存不能当作 no_grad 推理显存。没有固定正式实验的随机 seed，旧/新实验不保证权重逐值一致。\n',
        'Python 3.12.14、PyTorch 2.11.0+cu128、CUDA 12.8、Nsight Systems 2025.1.1、驱动 580.105.08；matmul TF32=False，float32 matmul precision=highest。版本、模型及测速源码 SHA256 和标注校验见 [environment.json](../outputs/nsight_run/environment.json)、[instrumentation_validation.json](../outputs/nsight_run/instrumentation_validation.json)。\n',
        'GPU span 是范围内第一个关联 GPU 操作开始到最后一个结束的跨度，包含中间空隙；kernel sum 是所有 kernel 持续时间之和，排除 GPU 空隙、CPU、同步等待和内存复制。Python wall 则还包含 CPU 发起操作和等待的经过时间；图中的它们是三个不同指标，不能相互替代。阶段没有额外插入同步，只有 step 边界同步；相邻阶段的 GPU span 不保证相加等于整步 span。[NVIDIA GPU projection 定义](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-sum-nvtx-gpu-projection-summary)。\n',
        '**反向线程的归属：**本轮 PyTorch autograd 在工作线程发起反向 CUDA kernel，外部 backward 标签在主线程，因此官方 NVTX GPU projection 表没有覆盖这些反向事件。阶段与 total 使用同一进程内、各 CPU 阶段范围中的 CUDA launch 时间归属（逐事件匹配 correlationId），保留工作线程；这适用于本次串行单步脚本，没有按 GPU 时间是否重叠猜归属。全部 kernel 的名称、次数、累计时间与官方全量 kernel 表一致，主线程阶段的 GPU span 与官方 projection 一致；具体差异存于 *_analysis_validation.json。\n',
        '采集只开启 CUDA/NVTX，measurement 范围排除初始化和 warm-up；attention 包含完整 MSA 的四次投影、RoPE、QKᵀ、mask、softmax、PV，attention_softmax 只包含手写 softmax 的 max/减/exp/sum/除。标签在测速脚本中临时包住原调用，退出后恢复；小规模 CUDA 输出和所有参数梯度加标签前后逐位相同，A1 算法文件没有修改。\n',
        '## 容量边界\n',table(cap_table),'\n',
        '长度按 2 倍增加至首个 OOM，small 从 1024 探至 2048，medium 在 1024 失败，512 的成功已有旧记录且本轮再次正式采集成功。表中 peak_allocated/peak_reserved 来自正式采集整个进程（包括初始化和预热）；二者不能相加，且不包含所有驱动/Profiler 开销，OOM 前峰值和异常栈另存在日志。\n',
        '## (a) 前向时间及旧 Python 对照\n',
        table(comparison[['model','seq_len','prior_python_mean_ms','nsight_gpu_span_mean_ms','kernel_sum_mean_ms']]),'\n',
        '![前向时间对照](../outputs/nsight_run/analysis/forward_comparison.png)\n',
        '三组柱分别表示旧 Python wall、Nsight GPU span、kernel sum；Python 与 span 的误差线表示 10 步标准差，kernel sum 的图中没有误差线。只有 S=512 有旧的同配置对照；本轮没有额外重跑旧 Python 基线，完整 [forward_comparison.csv](../outputs/nsight_run/analysis/forward_comparison.csv) 还保留标准差及 profiled_python 列，后者是在 profiler 内顺带记录的时间。\n',
        '## (b) 累计耗时最多的 kernel\n',table(hot_display),'\n',
        '完整模板名和各项时间见 [top_kernels.csv](../outputs/nsight_run/analysis/top_kernels.csv)。这里按 10 步的 Total Time 排名，calls_per_step 是总调用数除以 10；已逐 kernel 验证每步调用数相同。矩阵乘法调用数包含库的实际 kernel 分派，不能用 kernel 名称个数代替算法层面的矩阵乘法次数。[NVIDIA 统计字段说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。\n',
        '## (c) 主要非矩阵乘法 kernel\n',table(nonmat_display),'\n',
        'kernel 名在本表简写，完整 C++ 模板名保存在 [nonmatmul_kernels.csv](../outputs/nsight_run/analysis/nonmatmul_kernels.csv) 和 [kernel_observations.csv](../outputs/nsight_plan/kernel_observations.csv)。归属由 CUDA launch 的 correlationId 关联到 NVTX 范围确定，不能只凭 vectorized_elementwise 这个通用前缀判断。\n',
        'attention_softmax 中 exp/div/归约是 softmax 的组成部分；MSA 内、softmax 外的 scalar × tensor MulFunctor 与注意力缩放相符。MSA 外的 sigmoid 和每层两次 tensor × tensor 乘法与 SwiGLU 的 sigmoid、SiLU 乘法、门控乘法相符；后者依据标签排除和源码调用模式推断，没有额外给 FFN 打标签。其他共享逐元素 kernel 还用于 RoPE、RMSNorm 与残差，不能一概归入 FFN。[PyTorch 通用逐元素实现](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDALoops.cuh)。\n',
        '## (d) 训练阶段及整步矩阵乘法占比\n',
        '![small 各阶段占比](../outputs/nsight_run/analysis/small_matmul_share.png)\n',
        '![medium 各阶段占比](../outputs/nsight_run/analysis/medium_matmul_share.png)\n',
        '蓝色为矩阵乘法，橙色为该范围内其他 kernel；百分比的分母都是该行全部 kernel 的累计 GPU 时间。Forward only 是独立前向采集，Train: forward/loss/backward/AdamW 是训练的各阶段，Train: total 是整步；它们是不同范围，不能把各阶段百分比相加。阶段实际毫秒数与完整占比表见 [stage_observations.csv](../outputs/nsight_plan/stage_observations.csv)。\n',
        '## (e) 注意力 softmax 与矩阵乘法\n',table(at),'\n',
        '![注意力时间与运算量对照](../outputs/nsight_run/analysis/attention_comparison.png)\n',
        '左图是一次前向所有 MSA 层的 kernel 累计时间；右图蓝柱为 softmax/矩阵乘法时间比，橙柱为对应运算量比，纵轴为对数刻度，以便同时看清相差两个数量级的比值。完整 MSA 的矩阵乘法 FLOPs 为 L(8Bsd²+4Bs²d)，包括 QKV/输出投影和 QKᵀ、PV；没有把整个 attention 的 span 当成矩阵乘法时间。\n',
        '手写 softmax 每行长度 s：max 比较 s−1 次、减法 s 次、exp s 次、求和 s−1 次、除法 s 次，按每项一次标量操作计为 5s−2，总量 LBHs(5s−2)。max 比较严格说不属于 FLOP，exp/div 也不等于一条硬件指令，因此这里明示为近似运算量约定；若只统计加减乘除，softmax 的比值还会更低。这不影响 softmax 时间比例显著大于简单运算量比例的观察；内存读写、归约与多个 kernel 发起成本是合理解释，本次未单独测带宽或指令吞吐；[FlashAttention 论文](https://arxiv.org/abs/2205.14135)也强调应考虑注意力的内存访问成本，而不只看 FLOPs。\n',
        '## 每个配置的中文答题记录\n',
    ]
    for r in answers.to_dict('records'):
        report.append(f"### {r['model']} / S={r['seq_len']}\n")
        for key in 'abcde':report.append(f"**({key})** {r[key]}\n")
    report.extend(['## 原始记录与复现\n',
        '- [运行清单](../outputs/nsight_plan/experiment_manifest.csv)、[容量探测清单](../outputs/nsight_plan/capacity_probe_manifest.csv)、[容量结果](../outputs/nsight_plan/capacity_results.csv)。\n- [运行日志目录](../outputs/nsight_run/logs/)、[全部 CSV/图表](../outputs/nsight_run/analysis/)。每份 trace 的 *_validation.json 记录范围数量，*_analysis_validation.json 全量核对官方统计；*_events.csv 保留逐事件关联。\n- 表格同时导出 CSV、LaTeX、Typst；图同时导出 PNG、PDF。\n',
        '本轮的实验脚本为 [run_nsight_experiments.py](../run_nsight_experiments.py)，分析为 [analyze_nsight.py](../analyze_nsight.py)，报告与图表为 [report_nsight.py](../report_nsight.py)。GPU 采集已完成，不要直接重跑执行脚本覆盖现有文件；重新生成分析和报告可在 A2 目录执行 `uv run python analyze_nsight.py`、`uv run python report_nsight.py`。\n',
        '正式 trace：\n',paths,'\n'])
    DOC.write_text('\n'.join(report))
    print('Report:',DOC)
    print(comparison.to_string(index=False));print(at.to_string(index=False))


if __name__=='__main__':main()
