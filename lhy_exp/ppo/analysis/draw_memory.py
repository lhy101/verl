import re
from datetime import datetime
import matplotlib.pyplot as plt

def parse_log_file(filename):
    data_points = []
    pattern = re.compile(
        r'DEBUG:(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}):(.*?), memory allocated \(GB\): (\d+\.\d+), memory reserved \(GB\): (\d+\.\d+), device memory used/total \(GB\): (\d+\.\d+)'
    )
    
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                time_str = match.group(1)
                event = match.group(2).strip()
                allocated = float(match.group(3))
                reserved = float(match.group(4))
                total = float(match.group(5))
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                is_after = 'After' in event # 新增标识字段
                data_points.append({
                    'datetime': dt,
                    'event': event,
                    'allocated': allocated,
                    'reserved': reserved,
                    'total': total,
                    'is_after': is_after  # 记录是否为After事件
                })
    
    data_points.sort(key=lambda x: x['datetime'])
    if len(data_points) > 100:
        return data_points[:100]
    return data_points

def plot_memory_usage(data_points):
    if not data_points:
        print("No data points to plot.")
        return
    
    # 提取数据并分离After事件
    times = [p['seconds'] for p in data_points]
    events = [p['event'] for p in data_points]
    is_after = [p['is_after'] for p in data_points]
    
    # 分离不同事件的数据
    before_times = [t for t, after in zip(times, is_after) if not after]
    after_times = [t for t, after in zip(times, is_after) if after]
    
    before_alloc = [p['allocated'] for p in data_points if not p['is_after']]
    after_alloc = [p['allocated'] for p in data_points if p['is_after']]
    
    before_resv = [p['reserved'] for p in data_points if not p['is_after']]
    after_resv = [p['reserved'] for p in data_points if p['is_after']]
    
    before_total = [p['total'] for p in data_points if not p['is_after']]
    after_total = [p['total'] for p in data_points if p['is_after']]

    plt.figure(figsize=(100, 16))
    
    # 绘制基础折线
    alloc_line, = plt.plot(times, [p['allocated'] for p in data_points], 
                         color='blue', linestyle='-', label='Memory Allocated')
    resv_line, = plt.plot(times, [p['reserved'] for p in data_points],
                        color='orange', linestyle='-', label='Memory Reserved')
    total_line, = plt.plot(times, [p['total'] for p in data_points],
                        color='red', linestyle='-', label='Memory Total')
    
    # 添加不同标记（圆形 for Before，三角形 for After）
    plt.scatter(before_times, before_alloc, marker='o', s=80, color='blue', zorder=3)
    plt.scatter(after_times, after_alloc, marker='^', s=100, color='blue', zorder=3)
    plt.scatter(before_times, before_resv, marker='s', s=80, color='orange', zorder=3)
    plt.scatter(after_times, after_resv, marker='^', s=100, color='orange', zorder=3)
    plt.scatter(before_times, before_total, marker='x', s=80, color='red', zorder=3)
    plt.scatter(after_times, after_total, marker='^', s=100, color='red', zorder=3)
    
    # 仅标注Before事件
    y_max = max(max([p['allocated'] for p in data_points]), 
               max([p['reserved'] for p in data_points]),
               max([p['total'] for p in data_points]))
    vertical_offset = y_max * 0.12
    
    for p in data_points:
        if not p['is_after']:
            plt.annotate(
                p['event'],
                (p['seconds'], max(p['allocated'], p['reserved'], p['total'])),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                va='bottom',
                rotation=45,
                fontsize=8
            )
    
    # 时间差标注（保持原逻辑）
    for i in range(len(times)-1):
        t1, t2 = times[i], times[i+1]
        delta = t2 - t1
        mid_x = (t1 + t2)/2
        avg_y = (data_points[i]['allocated'] + data_points[i+1]['allocated'] +
                 data_points[i]['reserved'] + data_points[i+1]['reserved'] + 
                 data_points[i]['total'] + data_points[i+1]['total']) / 6
        
        plt.text(mid_x, avg_y + vertical_offset, 
                f"{delta:.1f}s", 
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.8))

    # 定制化图例
    from matplotlib.lines import Line2D
    legend_elements = [
        alloc_line,
        resv_line,
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Before Events'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
               markersize=10, label='After Events')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.xlabel('Time (seconds from first event)')
    plt.ylabel('GB')
    plt.title('Memory Usage Timeline')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("memory_new.png", dpi=400)

# 使用方式保持不变
filename = '/jizhicfs/lhy/workspace/verl/lhy_exp/Qwen_7b_1+31k_2nodes.log'
data_points = parse_log_file(filename)
if data_points:
    # 计算相对时间（保持原逻辑）
    base_time = data_points[0]['datetime']
    for p in data_points:
        p['seconds'] = (p['datetime'] - base_time).total_seconds()
    plot_memory_usage(data_points)
else:
    print("No valid data found.")