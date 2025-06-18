      
import os
import torch
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

def main():
    gpu_num = 8
    device_num = 1
    device_cnt = 0
    with tqdm(range(gpu_num)) as pbar:
        def map_func(gpu_id):
            sub_idx = gpu_id + device_cnt * gpu_num
            cmd = f'CUDA_VISIBLE_DEVICES={gpu_id % 8} python3 dreamlight_utils/preprocess_text_emb.py --sub_idx {sub_idx} --total {gpu_num * device_num}'
            os.system(command=cmd)
            pbar.update()

        pool = ThreadPool(gpu_num)
        pool.map(map_func, range(gpu_num))
        pool.close()

if __name__=='__main__':
    main()

    