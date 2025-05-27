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
            "old_output_ids", "cur_output_ids",
            "old_on_old_log_prob", "old_on_new_log_prob"
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing key {key} in {filename}")

        # 断言输入一致
        assert torch.equal(data["old_input_ids"], data["cur_input_ids"]), \
            f"Input mismatch in {filename}"
        
        old_lengths = []
        cur_lengths = []
        
        for old_seq, cur_seq, old_on_new_log_prob in zip(data["old_output_ids"], data["cur_output_ids"], data["old_on_new_log_prob"]):
            old_seq_len = (old_seq != 151643).sum().item()
            cur_seq_len = (cur_seq != 151643).sum().item()
            old_lengths.append(old_seq_len)
            cur_lengths.append(cur_seq_len)
            '''
            if cur_seq_len > 20000:
                print(cur_seq_len, old_on_new_log_prob)
            if cur_seq_len < 100:
                print(cur_seq_len, old_on_new_log_prob)
            '''

        old_on_old_log_prob = data["old_on_old_log_prob"].sum(dim=1) / torch.tensor(old_lengths)
        old_on_new_log_prob = data["old_on_new_log_prob"].sum(dim=1) / torch.tensor(cur_lengths)


        # 记录当前step的数据
        all_data.append({
            "step": int(filename.split("_")[1].split(".")[0]),
            "old_on_old_log_prob": old_on_old_log_prob.tolist(),
            "old_on_new_log_prob": old_on_new_log_prob.tolist(),
            "old_lengths": old_lengths,
            "cur_lengths": cur_lengths
        })

    return all_data

def visualize_results(all_data):
    # 准备绘图数据
    steps = []
    avg_old_on_old_log_prob = []
    avg_old_on_new_log_prob = []
    avg_log_prob_gap = []
    old_length_log_prob = {}
    cur_length_log_prob = {}
    max_length_log_prob_gap = {}
    
    # 遍历所有步骤
    for data in all_data:
        steps.append(data["step"])
        # 计算新旧log_prob的均值
        avg_old_on_old_log_prob.append(np.mean(data["old_on_old_log_prob"]))
        avg_old_on_new_log_prob.append(np.mean(data["old_on_new_log_prob"]))
        avg_log_prob_gap.append(np.mean(data["old_on_new_log_prob"]) - np.mean(data["old_on_old_log_prob"]))
        
        # 按长度分组旧数据
        for length, log_prob in zip(data["old_lengths"], data["old_on_old_log_prob"]):
            if length not in old_length_log_prob:
                old_length_log_prob[length] = []
            old_length_log_prob[length].append(log_prob)
        
        # 按长度分组新数据
        for length, log_prob in zip(data["cur_lengths"], data["old_on_new_log_prob"]):
            if length not in cur_length_log_prob:
                cur_length_log_prob[length] = []
            cur_length_log_prob[length].append(log_prob)

        # 按长度分组新数据
        for old_length, new_length, old_log_prob, new_log_prob in zip(data["old_lengths"], data["cur_lengths"], data["old_on_old_log_prob"], data["old_on_new_log_prob"]):
            if max(old_length, new_length) not in max_length_log_prob_gap:
                max_length_log_prob_gap[max(old_length, new_length)] = []
            max_length_log_prob_gap[max(old_length, new_length)].append(old_log_prob - new_log_prob)

    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 子图1：按step变化
    plt.subplot(1, 2, 1)
    # 绘制散点（新旧数据）
    for data in all_data:
        '''
        # 旧数据散点（蓝色）
        plt.scatter([data["step"]]*len(data["old_on_old_log_prob"]), 
                    data["old_on_old_log_prob"],
                    alpha=0.3, c='blue', s=10, 
                    label='Old' if data == all_data[0] else None)
        # 新数据散点（橙色）
        plt.scatter([data["step"]]*len(data["old_on_new_log_prob"]), 
                    data["old_on_new_log_prob"],
                    alpha=0.3, c='orange', s=10, 
                    label='Current' if data == all_data[0] else None)
        '''
        plt.scatter([data["step"]]*len(data["old_on_new_log_prob"]), 
                    np.array(data["old_on_new_log_prob"]) - np.array(data["old_on_old_log_prob"]),
                    alpha=0.3, c='orange', s=10, 
                    label='Current' if data == all_data[0] else None)
    
    # 绘制均值折线
    # plt.plot(steps, avg_old_on_old_log_prob, 'b-', linewidth=2, label='Old Average')
    # plt.plot(steps, avg_old_on_new_log_prob, 'r-', linewidth=2, label='Current Average')
    plt.plot(steps, avg_log_prob_gap, 'r-', linewidth=2, label='Current Average Gap')
    plt.xlabel("Training Step")
    plt.ylabel("Log Prob Gap")
    plt.title("Log Prob Gap vs Training Step")
    plt.grid(True)
    plt.legend()
    
    # 子图2：按长度变化
    plt.subplot(1, 2, 2)
    # 旧数据长度分布
    old_lengths = sorted(old_length_log_prob.keys())
    old_avg = [np.mean(old_length_log_prob[l]) for l in old_lengths]
    # 新数据长度分布
    cur_lengths = sorted(cur_length_log_prob.keys())
    cur_avg = [np.mean(cur_length_log_prob[l]) for l in cur_lengths]

    max_lengths = sorted(max_length_log_prob_gap.keys())
    gap_avg = [np.mean(max_length_log_prob_gap[l]) for l in max_lengths]
    
    # 绘制旧数据
    # plt.scatter(old_lengths, old_avg, alpha=0.5, c='blue', label='Old')
    # plt.plot(old_lengths, old_avg, 'b--', linewidth=1)
    # 绘制新数据
    # plt.scatter(cur_lengths, cur_avg, alpha=0.5, c='orange', label='Current')
    # plt.plot(cur_lengths, cur_avg, 'r--', linewidth=1)

    plt.scatter(max_lengths, gap_avg, alpha=0.5, c='orange', label='Current')
    plt.plot(max_lengths, gap_avg, 'r--', linewidth=1)
    
    plt.xlabel("Sequence Length")
    plt.ylabel("Log Prob")
    plt.title("Log Prob Gap vs Sequence Length")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("log_prob_gap_new.png", dpi=400)
    plt.close()

if __name__ == "__main__":
    analysis_data = analyze_pkl_files()
    visualize_results(analysis_data)