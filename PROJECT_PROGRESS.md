# PS-S6E2 项目进度记录

> 最后更新：2026-02-11 16:56:54 +08:00

## 1. 目标与当前状态

- 目标：冲刺 Kaggle `playground-series-s6e2` 前 20%
- 当前榜单规模：`2080` 条 public 排名
- 前 20% 阈值：约 `Rank 416`，分数约 `0.95366`
- 当前队伍：`Chica Chan`
- 当前成绩：`Rank 874`，`Public Score 0.95340`
- 与前 20% 分差：约 `+0.00026`

## 2. 本轮已完成工作（v10.1）

### 2.1 云端训练脚本优化

核心文件：`kaggle_kernel/ps_s6e2_gpu/train_gpu_catboost.py`

- 默认启用 XGB 分支（`--disable-xgb` 默认 `False`）
- 新增参数：
  - `--hybrid-alpha-step`
  - `--min-blend-weight`
- 新增融合候选搜索：
  - `tuned_rank_lgb_cat`
  - `tuned_rank_lgb_cat_xgb`
  - `hybrid_tuned_lgb_cat`
  - `hybrid_tuned_lgb_cat_xgb`
- 保留并强化三模型权重搜索：`tuned_lgb_cat_xgb`

### 2.2 自动化与校验工具

- 新增日常自动化脚本：`scripts/daily_online_loop.ps1`
  - 拉取 kernel 输出
  - 校验 submission 文件
  - 自动选出当天 2 个候选提交
- 新增本地轻量工具：`src/online_sprint.py`
  - `validate-submission`
  - `pick-daily`

### 2.3 本地最低性能验证

- 通过：`pytest tests/test_train_gpu_catboost_cloud.py tests/test_online_sprint.py tests/test_train_ranker.py tests/test_blend_submissions.py -q`
- 结果：`21 passed`

## 3. 云端实验与提交记录（最新）

- Kernel：`chicachan/ps-s6e2-optimized-hybrid-gpu`
- 版本：`v10`
- 状态：`KernelWorkerStatus.COMPLETE`
- 输出目录：`kaggle_outputs/cloud_v10`
- 关键产物：
  - `candidate_scores_cloud.csv`
  - `metrics_rank_push_cloud.json`
  - `submission_*.csv`

### 3.1 候选 CV Top3（OOF）

1. `hybrid_tuned_lgb_cat_xgb`：`0.9553032260`
2. `tuned_rank_lgb_cat_xgb`：`0.9553032115`
3. `tuned_lgb_cat_xgb`：`0.9553013873`

### 3.2 本轮已提交

1. `submission_hybrid_tuned_lgb_cat_xgb.csv` → `0.95337`
2. `submission_tuned_lgb_cat_xgb.csv` → `0.95337`

> 结论：本轮新增候选未超过当前最佳 `0.95340`。

## 4. 问题判断与迭代方向

当前现象：CV 持续提升但 Public 未同步提升，存在 **CV-LB 偏移/过拟合** 风险。

下一轮（v11）优先策略：

1. 收缩融合自由度：增大 `--min-blend-weight`（如 `0.10`）
2. 降低复杂融合占比：提高 `weighted_alpha`，减小纯 rank 依赖
3. 优先提交稳健候选：`safe / tuned_lgb_cat / tuned_lgb_cat_xgb`
4. 每轮仅 2 次计分提交，避免同质化提交

## 5. 下轮执行清单（可直接照做）

1. 推送新版本 kernel
2. 跑 `preset=full`，记录 `candidate_scores_cloud.csv`
3. 先用 `scripts/daily_online_loop.ps1` 选 2 个低相关候选
4. 提交后 15~30 分钟复盘 LB 变化
5. 更新本文件并沉淀“参数→结果”映射
