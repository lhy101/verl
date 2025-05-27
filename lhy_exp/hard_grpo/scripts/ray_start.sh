#!/bin/bash
source /jizhicfs/lhy/env/verl_H20.sh

HOSTFILE=${1:-"/jizhicfs/lhy/hosts/node03"}
PORT=8888                # Ray节点通信端口
DASHBOARD_PORT=8265      # Ray Dashboard端口

# 读取hostfile
mapfile -t hosts < "${HOSTFILE}"
if [ ${#hosts[@]} -eq 0 ]; then
    echo "Error: Empty hostfile"
    exit 1
fi

HEAD_IP=${hosts[0]}
workers=( "${hosts[@]:1}" )

# unset http_proxy && \
# unset https_proxy && \

# 启动Head节点
echo "Starting Head node at ${HEAD_IP}"
pssh -H "${HEAD_IP}" -i \
    "source /jizhicfs/lhy/env/verl_H20.sh && \
    ray start --head \
    --port=${PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --num-cpus=32"

# 启动Worker节点
if [ ${#workers[@]} -gt 0 ]; then
    echo "Starting ${#workers[@]} Worker nodes"
    pssh -H "${workers[*]}" -i \
        "source /jizhicfs/lhy/env/verl_H20.sh && \
        ray start --address=${HEAD_IP}:${PORT} \
        --num-cpus=32"
fi