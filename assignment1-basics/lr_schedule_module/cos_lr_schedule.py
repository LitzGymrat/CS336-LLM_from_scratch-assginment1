
import math
def lr_cosine_schedule(t:int,max_lr:float,min_lr:float,t_w:int,t_c:int):
    if t < t_w:
        return max_lr * (t/t_w)
    elif t_w <= t < t_c:
        a = math.pi*(t - t_w) /(t_c-t_w)

        return min_lr + (max_lr-min_lr)*(1 + math.cos(a)) / 2
    
    else:
        return min_lr
