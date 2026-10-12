import torch
def main():
    #原来的和累加的都采用f32:10.0001
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    #都采用f16:9.9531，差了0.05，精度最差
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    #输入 FP16，累计和 FP32:10.0021
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):

        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)


    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        #累加从f16转f32，和累加采用f16一致。
        s += x.type(torch.float32)
    print(s)
    #主要问题在于“accumulation的数据类型影响比较大”，如果是f32的话，即便累加的是已经被降级精度的tensor,最后的累加值也相对比较准确
    #如何理解？
    #数字都用二进制保存，但是很多数字无法用有限位二进制精确表示，所以导致只能取精度内的可取值，导致出现舍入的近似
    #用高精度保存积累，减少舍入带来的精度损失。
if __name__ == "__main__":
    main()