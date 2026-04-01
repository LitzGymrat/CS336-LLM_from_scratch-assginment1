#将pkl文件转换为npy文件的脚本
import numpy as np
import pickle

if __name__ == "__main__":
    path = "C:\\Users\\MilesZhu\\CS336-assignment1\\assignment1-basics\\tokenids_tinys_train.pkl"
    with open(path,"rb") as f:
        tokens_list = pickle.load(f)

    #转成npy文件
    tokens_np = np.array(tokens_list,dtype=np.uint16)

    np.save('tokenids_tinys_train.npy',tokens_np)
    print("已保存为npy文件")