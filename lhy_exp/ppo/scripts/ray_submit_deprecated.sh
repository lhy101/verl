# 提交作业（在本地执行）
echo "Submitting job to Ray cluster..."
ray job submit \
    --address="http://${HEAD_IP}:${DASHBOARD_PORT}" \
    --runtime-env=/jizhicfs/lhy/env/verl_H20.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \

echo "Job submitted to Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"