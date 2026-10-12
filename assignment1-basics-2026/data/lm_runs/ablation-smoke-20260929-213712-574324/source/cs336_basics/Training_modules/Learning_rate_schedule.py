import math
def get_lr_cosine_schedule(t:int,max_lr:float,min_lr:float,t_w:int,t_c:int):
    if t<t_w:
        lr_t = max_lr*(t/t_w)
    elif  t_w<=t<=t_c:
        lr_t = min_lr + (1+math.cos(math.pi*((t-t_w)/(t_c-t_w)))) * (max_lr - min_lr) / 2
    elif t>t_c:
        lr_t = min_lr
    return lr_t