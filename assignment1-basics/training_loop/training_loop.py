

from Transformer_lm.transformer_lm import Transformer_lm
from lr_schedule_module.cos_lr_schedule import lr_cosine_schedule
from gradient_clipping_module.gradient_clipping import gradient_clipping
from data_loader_module.get_batch import get_batch
from check_point_module.check_point_module import save_checkpoint

from cross_entropy_module.cross_entropy_loss import cross_entropy
from AdamW_module.AdamW import AdamW
import torch
import numpy as np
import os
import argparse
import wandb

@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_batches=10):
    model.eval()
    losses = torch.zeros(eval_batches)
    for k in range(eval_batches):
        X, Y = get_batch(data, batch_size, context_length, device)
        X, Y = X.to(device), Y.to(device)
        logits = model(X)
        loss = cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item() # 返回这 10 个 batch 的平均 Loss

def train():
    parser = argparse.ArgumentParser()
    # 实验名称，用于保存不同的 checkpoint
    parser.add_argument('--exp_name', type=str, default='baseline')
    # 是否开启架构创新
    parser.add_argument('--use_attn_res', action='store_true', help='开启 Attention Residuals')
    # 指定用哪张显卡 (0, 1, 2...)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    

    print(f"========== 开始实验: {args.exp_name} on GPU {args.gpu} ==========")
    #1.读取数据:使用memmap:不一次搬运完，需要用到的时候才搬运


    #先用硬编码。
    np_train_data = np.load("data/tokenids_tinys_train.npy",mmap_mode='r')
    #validation数据也读进来，用于eval评估
    np_valid_data = np.load("data/tokenids_tinys_valid.npy",mmap_mode='r')



    max_iters = 5000
    #2.设定超参数，初始化需要的实例
    vocab_size = 10000
    eval_iters = 100
    checkpoint_iters = 500
    context_length=256
    d_model=512
    num_layers=4
    num_heads=16
    d_ff=1344
    rope_theta=10000
    device='cuda' if torch.cuda.is_available() is True else 'cpu'
    dtype=torch.float32
    batch_size = 64
    lr = 1e-3
    #学习率规划：先用经验值
    max_lr = 1e-3
    min_lr = 1e-4
    warmup_iters = 10
    cosine_end_iters = max_iters
    betas = (0.9,0.99)
    eps = 1e-5
    weight_decay = 0.01
    model = Transformer_lm(vocab_size,context_length,d_model,num_layers,
                                    num_heads,d_ff,rope_theta,device = device,dtype=dtype)#.to(device) #记得搬到设备上
    model = model.to(device)
                    #把参数交给优化器（操作的就是设备上的参数，不用再指定）
    optimizer = AdamW(model.parameters(),lr,betas,eps,weight_decay)
    gradient_clip = True
    max_l2_norm = 1
    lr_schedule = True
    
    wandb.init(
        project="CS336-LLM-Project", # 项目名称（网页上显示的文件夹名）
        name=args.exp_name,         # 这次实验的名字
        config={
            "batch_size": batch_size,
            "learning_rate": lr,
            "max_iters": max_iters
        }
    )
    #data = Data_loader(np_data,b_size = batch_size,s_len=context_length,device=device)
    #data:(input,target)
    #其中二者都是(batch_size,context_length)
    #3.迭代训练
    for iter in range(max_iters):
                    #每次迭代，取一次数据
        x,y = get_batch(np_train_data,b_size = batch_size,s_len=context_length,device=device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = cross_entropy(logits,y)

        #计算损失前优化器先把上次梯度清零
        optimizer.zero_grad()

        loss.backward()


        #梯度裁剪：进优化器之前。
        if gradient_clip is True:
            gradient_clipping(model.parameters(),max_l2_norm=max_l2_norm)

        optimizer.step()

        
        if lr_schedule is True:
            #更新学习率
            new_lr = lr_cosine_schedule(iter+1,max_lr,min_lr,warmup_iters,cosine_end_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": current_lr,
            "iter": iter
        })
        


    #4.五步走：获取数据，前向传播，计算损失，反向传播，梯度下降
    #可选的学习率规划和梯度截断

    #5.打印日志，以及存档点
        if (iter+1) % eval_iters == 0:
            #验证集评估：开启评估模式，关闭梯度记录
                valid_loss = estimate_loss(model,np_valid_data,batch_size,context_length,device,eval_batches=10)
                train_loss = estimate_loss(model,np_train_data,batch_size,context_length,device,eval_batches=10)
                #loss是张量类型，需要取出数值
                print(f"第{iter+1}轮:训练损失为{loss.item():.3f}\n")
                print(f"验证集平均损失为{valid_loss:.3f}\n")
                print(f"训练集平均损失为{train_loss:.3f}\n")
                
                wandb.log({
                    "val/loss": valid_loss,
                    "train/avg_loss":train_loss,
                    "iter": iter+1
                })
            
        if (iter+1)% checkpoint_iters == 0:
            save_dir = os.path.join("outputs", args.exp_name, "checkpoints")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ckpt_step_{iter+1}.pt")
            
            save_checkpoint(model,optimizer,iter+1,save_path)

            print(f"第{iter+1}轮，存档至{save_path}\n")


if __name__ == "__main__":
    train()