import os
import time
from datetime import datetime
from collections import defaultdict
from contextlib import contextmanager
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from datetime import datetime
from typing import Optional, Dict, Any

@contextmanager
def _torch_profiler(
    file_prefix: str = None,
    profile_time: bool = True,
    profile_trace: bool = False,
    activities: Optional[list] = None,
    profiler_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Torch Profiler上下文管理器，支持自定义文件名前缀
    
    参数：
    file_prefix : 可选的文件路径名前缀（例如"policy_update"）
    time : 是否记录被包裹代码的执行时间
    trace : 是否启用PyTorch的性能跟踪
    activities : 要监控的活动列表（默认监控CUDA）
    profiler_kwargs : 传递给torch.profiler.profile的额外参数
    """
    if "verl_torch_profile" not in os.environ:
        yield None
        return

    trace_dir = os.path.abspath(os.environ["verl_torch_profile"])
    os.makedirs(trace_dir, exist_ok=True)

    # 生成文件名组件
    host_name = os.environ.get("LOCAL_IP", "localhost")
    cuda_id = os.environ.get("RAY_LOCAL_RANK", "0")
    final_prefix = f"{host_name}_cuda:{cuda_id}"
    if file_prefix:
        final_prefix = f"{file_prefix}_{final_prefix}"

    start_time = None
    if profile_time:
        start_time = time.perf_counter()
        start_date_time = datetime.now().time()

    def _trace_handler(prof: profile):
        """处理跟踪数据的回调函数"""
        try:
            prof.export_chrome_trace(f"{trace_dir}/{final_prefix}.json")
        except Exception as e:
            print(f"导出计算时间线失败: {str(e)}")
        try:
            prof.export_memory_timeline(f"{trace_dir}/{final_prefix}_memory.html")
        except Exception as e:
            print(f"导出内存时间线失败: {str(e)}")

    try:
        if profile_trace:
            if activities is None:
                activities = [ProfilerActivity.CUDA]
            prof_args = {
                "activities": activities,
                "schedule": schedule(wait=0, warmup=0, active=1, repeat=1),
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": True,
                "on_trace_ready": _trace_handler,
            }
            if profiler_kwargs:
                prof_args.update(profiler_kwargs)
            with profile(**prof_args) as prof:
                yield prof
        else:
            yield None
    finally:
        if profile_time:
            end_time = time.perf_counter()
            end_data_time = datetime.now().time()
            execution_time = end_time - start_time
            time_filename = os.path.join(trace_dir, f"{final_prefix}_time.txt")
            with open(time_filename, 'w') as f:
                f.write(f"{file_prefix} execution_time: {execution_time}s\n")
                f.write(f"{file_prefix} start date time: {start_date_time}\n")
                f.write(f"{file_prefix} end date time: {end_data_time}")