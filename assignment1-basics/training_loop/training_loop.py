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


def train():
    parser = argparse.ArgumentParser()
    # 实验名称，用于保存不同的 checkpoint
    parser.add_argument('--exp_name', type=str, default='baseline')
    # 是否开启架构创新
    parser.add_argument('--use_attn_res', action='store_true', help='开启 Attention Residuals')
    # 指定用哪张显卡 (0, 1, 2...)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    # 核心：让 PyTorch 只看到你分配给它的那张卡！
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"========== 开始实验: {args.exp_name} on GPU {args.gpu} ==========")
    #1.读取数据:使用memmap:不一次搬运完，需要用到的时候才搬运


    #先用硬编码。
    np_train_data = np.load("assignment1-basics/tokenids_tinys_train.pkl",mmap_mode='r')
    #validation数据也读进来，用于eval评估
    np_valid_data = np.load("assignment1-basics/tokenids_tinys_valid.pkl",mmap_mode='r')



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
    batch_size = 32
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
    
    #data = Data_loader(np_data,b_size = batch_size,s_len=context_length,device=device)
    #data:(input,target)
    #其中二者都是(batch_size,context_length)
    #3.迭代训练
    for iter in range(max_iters):
                    #每次迭代，取一次数据
        x,y = get_batch(np_train_data,b_size = batch_size,s_len=context_length,device=device)

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
        


    #4.五步走：获取数据，前向传播，计算损失，反向传播，梯度下降
    #可选的学习率规划和梯度截断

    #5.打印日志，以及存档点
        if iter % eval_iters == 0:
            x_valid,y_valid = get_batch(np_valid_data,b_size = batch_size,s_len=context_length,device=device)
            logits_valid = model(x_valid)
            loss_valid = cross_entropy(logits_valid,y_valid)
            #loss是张量类型，需要取出数值
            print(f"第{iter+1}轮:训练损失为{loss.item():.3f}\n")
            print(f"验证集损失为{loss_valid.item():.3f}\n")
        if iter % checkpoint_iters == 0:
            save_dir = "outputs/baseline/checkpoints"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"ckpt_step_{iter+1}.pt")
            
            save_checkpoint(model,optimizer,iter+1,save_path)

            print(f"第{iter+1}轮，存档至{save_path}\n")


if __name__ == "__main__":
    train()