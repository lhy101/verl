import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(file_path):
    data = []
    required_keys = [
        'global_seqlen/mean',
        'global_seqlen/balanced_max',
        'response_length/mean',
        'response_length/max',
        'timing_s/gen',
        'timing_s/update_critic',
        'timing_s/update_actor'
    ]
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            entries = line.split(' - ')
            entry_dict = {}
            for entry in entries:
                if ':' in entry:
                    key, value = entry.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    try:
                        entry_dict[key] = float(value)
                    except:
                        continue
            
            if all(key in entry_dict for key in required_keys):
                data.append(entry_dict)
    
    return data

def plot_subfigures(data):
    x_keys = [
        'global_seqlen/mean',
        'global_seqlen/balanced_max',
        'response_length/mean',
        'response_length/max'
    ]
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Timing Metrics Analysis', fontsize=14)
    
    colors = {'gen': 'blue', 'update_critic': 'orange', 'update_actor': 'green'}
    
    for idx, x_key in enumerate(x_keys):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        
        # Collect data
        x_values = [d[x_key] for d in data]
        y_gen = [d['timing_s/gen'] for d in data]
        y_critic = [d['timing_s/update_critic'] for d in data]
        y_actor = [d['timing_s/update_actor'] for d in data]
        
        # Group data and calculate averages
        groups = {}
        for x, g, c, a in zip(x_values, y_gen, y_critic, y_actor):
            if x not in groups:
                groups[x] = {'gen': [], 'update_critic': [], 'update_actor': []}
            groups[x]['gen'].append(g)
            groups[x]['update_critic'].append(c)
            groups[x]['update_actor'].append(a)
        
        sorted_x = sorted(groups.keys())
        avg_gen = [np.mean(groups[x]['gen']) for x in sorted_x]
        avg_critic = [np.mean(groups[x]['update_critic']) for x in sorted_x]
        avg_actor = [np.mean(groups[x]['update_actor']) for x in sorted_x]
        
        # Plot scatters
        ax.scatter(x_values, y_gen, c=colors['gen'], alpha=0.4, 
                  label='gen' if idx == 0 else "", s=20)
        ax.scatter(x_values, y_critic, c=colors['update_critic'], alpha=0.4,
                  label='update_critic' if idx == 0 else "", s=20)
        ax.scatter(x_values, y_actor, c=colors['update_actor'], alpha=0.4,
                  label='update_actor' if idx == 0 else "", s=20)
        
        # Plot average lines
        ax.plot(sorted_x, avg_gen, color=colors['gen'], linewidth=2, zorder=3)
        ax.plot(sorted_x, avg_critic, color=colors['update_critic'], linewidth=2, zorder=3)
        ax.plot(sorted_x, avg_actor, color=colors['update_actor'], linewidth=2, zorder=3)
        
        ax.set_xlabel(x_key, fontsize=10)
        ax.set_ylabel('Time (seconds)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.88, 0.12),
              frameon=True, fontsize=10, title='Metrics')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("time_relation_32b.png", dpi=400)

# 使用示例
data = parse_log_file('/jizhicfs/lhy/workspace/verl/lhy_exp/hard_ppo/Qwen_32b_sync.log')
plot_subfigures(data)