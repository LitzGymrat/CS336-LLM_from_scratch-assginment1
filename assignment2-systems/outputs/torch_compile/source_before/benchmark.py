import torch
import argparse
import timeit
import statistics
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx
from cs336_basics.Transformer.Transformer_lm import Transformer_lm
from cs336_basics.Training_modules.AdamW import AdamW
from cs336_basics.Training_modules.Crossentropyloss import cross_entropy
from cs336_systems.reporting import save_table
from cs336_systems.attention_profiling import attention_ranges
from cs336_systems.memory_profiling import profile_memory, begin_memory_measurement
def parse_args():
    #命令行参数读取
    #创建parser
    parser = argparse.ArgumentParser()
    #模型参数
    parser.add_argument("--d_model",type = int,default = 32)
    parser.add_argument("--vocab_size",type = int,default = 64)
    parser.add_argument("--d_ff",type = int,default = 128)
    parser.add_argument("--num_layers",type = int,default = 1)
    parser.add_argument("--num_heads",type = int,default = 4)
    parser.add_argument("--rope_theta",type = float,default = 10000.0)
    parser.add_argument("--ffn_type",choices=["swiglu","silu"],default = "swiglu")
    parser.add_argument("--norm_position",choices=["pre","post"],default = "pre")
    parser.add_argument("--use_rope",action=argparse.BooleanOptionalAction,default = True)
    #输入参数
    parser.add_argument("--context_len",type = int,default = 8)
    parser.add_argument("--batch_size",type = int,default = 2)
    #运行
    parser.add_argument("--device",choices=["cpu","cuda"],default = "cpu")
    parser.add_argument("--precision",choices=["fp32","bf16"],default = "fp32")
    parser.add_argument("--seed",type = int,default = 0)
    #优化器
    parser.add_argument("--lr",type = float,default = 0.001)
    #测量
    parser.add_argument("--steps",type = int,default = 3)
    parser.add_argument("--warmup",type = int,default = 2) #type:将输入字符串转成所需类型
    parser.add_argument("--mode",choices=["forward","forward_backward","train","train_stages"],default = "forward")
    parser.add_argument("--output_prefix", default="outputs/benchmark")
    parser.add_argument("--nvtx",action=argparse.BooleanOptionalAction,default = False)
    parser.add_argument("--memory_profile",action=argparse.BooleanOptionalAction,default = False,
                        help="记录 CUDA 显存历史，输出 .pickle 和 .memory.json")
    parser.add_argument("--inference",action=argparse.BooleanOptionalAction,default = False,
                        help="仅用于 forward：关闭梯度记录，测量推理")
    
    args = parser.parse_args() #读取参数
    return args #返回参数

#将选择NVTX或空context提取成一个小函数
def profiler_range(name:str,use_nvtx:bool,device):
    if use_nvtx and device.type == "cuda":
        return nvtx.range(name)
    return nullcontext()


def precision_context(precision:str,device):
    #参数仍是 FP32；开启时由 autocast 为各个运算选择精度。
    if precision == "bf16":
        return torch.autocast(device_type=device.type,dtype=torch.bfloat16)
    return nullcontext()


#写一个专门用于执行指定模式操作的函数。
def run_step(model:Transformer_lm,input:torch.Tensor,target:torch.Tensor,mode:str,optimizer:torch.optim.Optimizer|None,use_nvtx:bool = False,precision:str = "fp32",inference:bool = False):
    match mode:
        case "train":
            #先清空梯度
            optimizer.zero_grad()
            with precision_context(precision,input.device):
                with profiler_range("forward",use_nvtx,input.device):
                    out_logits = model.forward(input)

                with profiler_range("loss",use_nvtx,input.device):
                    loss = cross_entropy(out_logits,target)
                
            with profiler_range("backward",use_nvtx,input.device):
                loss.backward()
            with profiler_range("optimizer",use_nvtx,input.device):            
                optimizer.step()
        case "forward":
            grad_context = torch.no_grad() if inference else nullcontext()
            with grad_context, precision_context(precision,input.device), profiler_range("forward",use_nvtx,input.device):
                out_logits = model.forward(input)
        case "forward_backward":
            #避免梯度累积，反向之前要清空梯度
            model.zero_grad()
            with precision_context(precision,input.device):
                with profiler_range("forward",use_nvtx,input.device):
                    out_logits = model.forward(input)
                with profiler_range("loss",use_nvtx,input.device):
                    loss = cross_entropy(out_logits,target)
#   
            with profiler_range("backward",use_nvtx,input.device):
                loss.backward()
#写一个专门计时的函数
def measure_call(model:Transformer_lm,input:torch.Tensor,target:torch.Tensor,mode:str,steps:int,optimizer:torch.optim.Optimizer|None,device,use_nvtx:bool = False,precision:str = "fp32",inference:bool = False):
    call_times = []
    for _ in range(steps):
        if device.type  == "cuda":
            torch.cuda.synchronize() #开启一次等待：等待之前提交的GPU工作完成
        start = timeit.default_timer()
        #在不同的需要profiler的区间中区分标注
        with profiler_range("step",use_nvtx,input.device):
            run_step(model,input,target,mode,optimizer,use_nvtx,precision,inference)
            if device.type  == "cuda":
                #每个step后也同步：返回调用后等待GPU工作完成
                torch.cuda.synchronize()  
        end =   timeit.default_timer()
        elapsed_ms = (end-start) * 1000
        call_times.append(elapsed_ms)
    return call_times

def measure_train_stages(model:Transformer_lm,input:torch.Tensor,target:torch.Tensor,mode:str,steps:int,optimizer:torch.optim.Optimizer|None,device,precision:str = "fp32"):
    forward_times = []
    backward_times = []
    optimizer_times = []
    for _ in range(steps):
        if device.type  == "cuda":
            torch.cuda.synchronize() #开启一次等待：等待之前提交的GPU工作完成
        
        optimizer.zero_grad()
        forward_start = timeit.default_timer()

        with precision_context(precision,device):
            out_logits = model.forward(input)
            loss = cross_entropy(out_logits,target)
        if device.type  == "cuda":
            #每个step后也同步：返回调用后等待GPU工作完成
            torch.cuda.synchronize()  
        forward_end = timeit.default_timer()
        forward_times.append((forward_end - forward_start)*1000)
        if device.type  == "cuda":
            #每个step后也同步：返回调用后等待GPU工作完成
            torch.cuda.synchronize()  
        backward_start = timeit.default_timer()

        loss.backward()
        if device.type  == "cuda":
            #每个step后也同步：返回调用后等待GPU工作完成
            torch.cuda.synchronize()  
        backward_end = timeit.default_timer()
        backward_times.append((backward_end - backward_start)*1000)

        if device.type  == "cuda":
            #每个step后也同步：返回调用后等待GPU工作完成
            torch.cuda.synchronize()  
        optimizer_start = timeit.default_timer()
        optimizer.step()
        
        if device.type  == "cuda":
            #每个step后也同步：返回调用后等待GPU工作完成
            torch.cuda.synchronize()  
        optimizer_end = timeit.default_timer()
        optimizer_times.append((optimizer_end - optimizer_start)*1000)

    return forward_times,backward_times,optimizer_times

def benchmark(vocab_size:int,d_model:int, num_heads:int,d_ff:int,num_layers:int,context_len:int,rope_theta:float,batch_size:int,lr:float,steps:int,warmup:int,mode:str,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 norm_position:str = "pre", use_rope:bool = True, ffn_type:str = "swiglu",output_prefix: str = "outputs/benchmark",use_nvtx:bool = False,precision:str = "fp32",seed:int = 0,memory_profile:bool = False,inference:bool = False):

    if memory_profile and device.type != "cuda":
        raise ValueError("CUDA 显存快照需要 --device cuda")
    if inference and mode != "forward":
        raise ValueError("--inference 只与 --mode forward 一起使用")
    torch.manual_seed(seed)
    model = Transformer_lm(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        context_len=context_len,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
        norm_position=norm_position,
        use_rope=use_rope,
        ffn_type=ffn_type,
    )
    if inference:
        model.eval()
    if mode in ("train", "train_stages"):
        optimizer = AdamW(
            params=model.parameters(),
            lr = lr
    )
    else:
        optimizer = None
        #准备随机数据:搬运到相应设备
    rand_input = torch.randint(0,vocab_size,(batch_size,context_len),device = device)
    rand_target = torch.randint(0,vocab_size,(batch_size,context_len),device = device)
    
    memory_context = profile_memory(output_prefix, device) if memory_profile else nullcontext({})
    with memory_context as memory_result:
        #预热：阶段测量同样用完整训练步骤，建立优化器状态。
        warmup_mode = "train" if mode == "train_stages" else mode
        for _ in range(warmup):
            if device.type == "cuda":
                torch.cuda.synchronize()
            run_step(model,rand_input,rand_target,warmup_mode,optimizer,precision=precision,inference=inference)
            if device.type == "cuda":
                torch.cuda.synchronize()

        if memory_profile:
            begin_memory_measurement(memory_result, device)

        #正式计时；显存快照的正常记录范围也是这里。
        with attention_ranges(model, use_nvtx and device.type == "cuda"), profiler_range("measurement",use_nvtx,device):
            if mode == "train_stages":
                forward_times, backward_times, optimizer_times = measure_train_stages(
                    model, rand_input, rand_target, mode, steps, optimizer, device, precision
                )
                stage_times = {
                    "forward_with_loss": forward_times,
                    "backward": backward_times,
                    "optimizer": optimizer_times,
                }
            else:
                times = measure_call(model,rand_input,rand_target,mode,steps,optimizer,device,use_nvtx,precision,inference)
                stage_times = {"total": times}

    record = {
    "vocab_size": vocab_size,
    "d_model": d_model,
    "d_ff": d_ff,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "rope_theta": rope_theta,
    "batch_size": batch_size,
    "seq_len": context_len,
    "device": str(device),
    "precision": str(torch.bfloat16) if precision == "bf16" else str(dtype),
    "parameter_dtype": str(next(model.parameters()).dtype),
    "autocast": precision == "bf16",
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "seed": seed,
    "norm_position": norm_position,
    "use_rope": use_rope,
    "ffn_type": ffn_type,
    "lr": lr,
    "mode": mode,
    "warmup": warmup,
    "steps": steps,
    "memory_profile": memory_profile,
    "inference": inference,
}
    record.update(memory_result)

    records = []
    for stage, times in stage_times.items():
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times)
        stage_record = record.copy()
        stage_record["stage"] = stage
        stage_record["mean_ms"] = mean_time
        stage_record["std_ms"] = std_time
        records.append(stage_record)

        print(f"[{stage}] 每次耗时(ms):", times)
        print(f"平均耗时：{mean_time:.3f} ms")
        print(f"标准差：{std_time:.3f} ms")

    save_table(records, output_prefix)
    

def main():
    args = parse_args()
    print(args)
    benchmark(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        context_len=args.context_len,
        rope_theta=args.rope_theta,
        norm_position=args.norm_position,
        use_rope=args.use_rope,
        ffn_type=args.ffn_type,
        device=torch.device(args.device),
        dtype=torch.float32,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        warmup=args.warmup,
        mode=args.mode, 
        output_prefix = args.output_prefix,
        use_nvtx = args.nvtx,
        precision = args.precision,
        seed = args.seed,
        memory_profile = args.memory_profile,
        inference = args.inference,
    )

if __name__ == "__main__" :
    main()
