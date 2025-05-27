import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def analyze_pkl_files(folder_path="/root/gen_samples"):
    # 收集所有数据
    all_data = []
    
    # 读取目录下所有pkl文件
    pkl_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    pkl_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))  # 按step排序
    
    # 遍历所有文件
    for filename in tqdm(pkl_files, desc="Processing files"):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            
        # 验证数据完整性
        required_keys = [
            "old_input_ids", "cur_input_ids",
            "old_output_ids", "cur_output_ids"
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key {key} in {filename}")

        # 断言输入一致
        assert torch.equal(data["old_input_ids"], data["cur_input_ids"]), \
            f"Input mismatch in {filename}"

        # 计算差异指标
        batch_ratios = []
        batch_lengths = []
        
        for old_seq, cur_seq in zip(data["old_output_ids"], data["cur_output_ids"]):
            old_seq_len = (old_seq != 151643).sum().item()
            cur_seq_len = (cur_seq != 151643).sum().item()
            min_len = min(old_seq_len, cur_seq_len)
            diff_pos = None
            
            # 寻找第一个差异位置
            for i in range(min_len):
                if old_seq[i] != cur_seq[i]:
                    diff_pos = i
                    break
            
            # 处理完全一致但长度不同的情况
            if diff_pos is None:
                diff_pos = min_len  # 视为在最短长度处产生差异
                
            ratio = diff_pos / min_len
            batch_ratios.append(ratio)
            batch_lengths.append(min_len)

        # 记录当前step的数据
        all_data.append({
            "step": int(filename.split("_")[1].split(".")[0]),
            "ratios": batch_ratios,
            "lengths": batch_lengths
        })

    return all_data

def visualize_results(all_data):
    # 准备绘图数据
    steps = []
    avg_ratios = []
    length_ratios = {}
    
    # 遍历所有步骤
    for data in all_data:
        # 计算步骤平均值
        steps.append(data["step"])
        avg_ratios.append(np.mean(data["ratios"]))
        
        # 按长度分组
        for length, ratio in zip(data["lengths"], data["ratios"]):
            if length not in length_ratios:
                length_ratios[length] = []
            length_ratios[length].append(ratio)

    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 子图1：按step变化
    plt.subplot(1, 2, 1)
    # 绘制散点
    for data in all_data:
        plt.scatter([data["step"]]*len(data["ratios"]), data["ratios"],
                   alpha=0.3, c='blue', s=10)
    # 绘制均值折线
    plt.plot(steps, avg_ratios, 'r-', linewidth=2, label='Average')
    plt.xlabel("Training Step")
    plt.ylabel("Divergence Position Ratio")
    plt.title("Divergence Ratio vs Training Step")
    plt.grid(True)
    plt.legend()
    
    # 子图2：按长度变化
    plt.subplot(1, 2, 2)
    lengths = sorted(length_ratios.keys())
    avg_length_ratios = [np.mean(length_ratios[l]) for l in lengths]
    
    plt.scatter(lengths, avg_length_ratios, alpha=0.5, c='green')
    plt.plot(lengths, avg_length_ratios, 'b--', linewidth=1)
    plt.xlabel("Sequence Length (min)")
    plt.ylabel("Average Divergence Ratio")
    plt.title("Divergence Ratio vs Sequence Length")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("gen_samples.png", dpi=400)

if __name__ == "__main__":
    analysis_data = analyze_pkl_files()
    visualize_results(analysis_data)