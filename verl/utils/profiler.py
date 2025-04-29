import os
import shutil
from collections import defaultdict
from contextlib import contextmanager
from torch.profiler import profile, ProfilerActivity, schedule, record_function
from datetime import datetime
from typing import Optional, Dict, Any

@contextmanager
def _torch_profiler(
    file_prefix: str = None,
    profile_time: bool = True,
    activities: Optional[list] = None,
    profiler_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Torch Profiler上下文管理器，支持自定义文件名前缀
    
    参数：
    file_prefix : 可选的文件路径名前缀（例如"policy_update"）
    activities : 要监控的活动列表（默认监控CPU和CUDA）
    profiler_kwargs : 传递给torch.profiler.profile的额外参数
    """

    if "verl_torch_profile" not in os.environ:
        yield

    trace_dir = os.path.abspath("/jizhicfs/trace")
    # if os.path.exists(trace_dir):
        # shutil.rmtree(trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    # 生成文件名组件
    host_name = os.environ.get("LOCAL_IP", "localhost")
    cuda_id =  os.environ.get("RAY_LOCAL_RANK", "0")
    
    # 设置默认activities
    if activities is None:
        activities = [
            ProfilerActivity.CUDA,
        ]
    
    # 合并profiler参数
    prof_args = {
        "activities": activities,
        "schedule": schedule(wait=0, warmup=0, active=1, repeat=1),
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": True,
        "on_trace_ready": None  # 稍后定义
    }
    if profiler_kwargs:
        prof_args.update(profiler_kwargs)

    # 定义trace处理回调
    def _trace_handler(prof: profile):
        """处理跟踪数据的回调函数"""
        # 每次生成跟踪时创建新时间戳
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        
        # 构造最终文件名前缀
        # final_prefix = f"{host_name}_cuda:{cuda_id}_{timestamp}"
        final_prefix = f"{host_name}_cuda:{cuda_id}"
        if file_prefix:
            final_prefix = f"{file_prefix}_{final_prefix}"

        # 导出跟踪数据
        if profile_time:
            try:
                prof.export_chrome_trace(f"{trace_dir}/{final_prefix}.json")
            except Exception as e:
                print(f"导出计算时间线失败: {str(e)}")
        try:
            prof.export_memory_timeline(f"{trace_dir}/{final_prefix}_memory.html")
        except Exception as e:
            print(f"导出内存时间线失败: {str(e)}")

    # 更新profiler参数中的回调
    prof_args["on_trace_ready"] = _trace_handler

    # 创建profiler实例
    with profile(**prof_args) as prof:
        try:
            yield prof
        finally:
            # 确保所有数据刷新
            prof.stop()