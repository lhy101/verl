import os
from collections import defaultdict

def process_files(folder_path):
    # 初始化字典，用于存储分类后的数据
    result_dict = defaultdict(list)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                line = file.readline().strip()
                # 分割键值对
                key, value_part = line.split(':')
                key = key.strip()
                # 提取数值并转换类型
                value = float(value_part.strip().rstrip('s'))
                result_dict[key].append(value)
    
    # 对每个键对应的列表进行排序
    for key in result_dict:
        result_dict[key].sort()
    
    # 转换为普通字典并打印
    print(dict(result_dict))

# 使用示例，替换为你的文件夹路径
folder_path = '/jizhicfs/trace'
process_files(folder_path)