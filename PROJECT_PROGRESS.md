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


## 6. 最新执行结果（2026-02-11）

### 6.1 云端训练 v11（meta gate）
- Kernel 版本：`v11`（`chicachan/ps-s6e2-optimized-hybrid-gpu`）
- 关键参数：`meta_min_gain=2e-5`
- 候选 CV：
  1. `aggressive`：`0.9552946`
  2. `tuned_lgb_cat`：`0.9552937`
  3. `safe`：`0.9552807`
- 提交：`submission_best.csv`
- Public LB：`0.95335`（未超过当前最好）

### 6.2 锚定混合（低风险）
- 方案：`0.90 * v6_aggressive_rebuilt + 0.10 * v11_best`
- 提交文件：`submission_anchor_blend_90_10.csv`
- Public LB：`0.95340`
- 结论：追平历史最好分（稳定但未突破）

### 6.3 云端训练 v12（更严格门控）
- Kernel 版本：`v12`
- 关键参数：`meta_min_gain=3e-5`，`min_blend_weight=0.10`
- 最优候选：`aggressive`（CV `0.9552944`）
- 提交文件：`submission_aggressive.csv`
- Public LB：`0.95335`
- 结论：继续验证到 CV-LB 偏移，单纯加严门控暂未带来提升

### 6.4 当前最优与下一步
- 当前最佳 Public LB 仍为：`0.95340`
- 当前最稳健提交：`submission_aggressive_rebuilt.csv` 与 `submission_anchor_blend_90_10.csv`（同分）
- 下一轮建议：
  1. 固定 `v6` 为锚，仅在 `0.92~0.98` 小权重区间做 1-2 次线上探索
  2. 云端脚本只保留 `safe` + `tuned_lgb_cat` 两候选，减少同质候选干扰
  3. 若仍无提升，切到特征侧增益（统计聚合/分箱交叉）再开新轮


## 7. v13 新冲榜思路（参考优秀Kernel后落地）

### 7.1 参考来源（Kaggle）
- `achilov15/corrprune-median-blend-0-95397`：相关性剪枝 + 中位数融合
- `azzamradman/0226-blend-the-blender`：分位数融合（P25）
- `dmahajanbe23/feature-growth-catboost-xgb-ordered-boosting`：目标统计特征 + 交叉特征扩展

### 7.2 本仓库落地改动
- 云端脚本新增 **Cross-Fit Target Stats**（无泄漏）：
  - 为低基数列生成 `te_*` + `freq_*`，并构建少量 `te` 交叉特征
  - 默认新增特征数：30（13列来源）
- 云端脚本新增 **Corr-Prune Quantile 候选**：
  - `robust_median`（q=0.50）
  - `robust_p25`（q=0.25）
- 保留现有 meta-gate 流程，避免 OOF 过拟合挑选

### 7.3 云端结果（2026-02-11）
- Kernel 版本：`v13`
- 关键候选 CV：
  - `aggressive`: `0.9555916`
  - `tuned_lgb_cat`: `0.9555904`
  - `robust_median`: `0.9555795`
  - `robust_p25`: `0.9555782`
- 提交结果：
  1. `submission_aggressive.csv` -> `0.95374`
  2. `submission_robust_median.csv` -> `0.95374`

### 7.4 当前结论
- 新方案已**显著突破**历史最好 `0.95340`，当前提升到 `0.95374`（+0.00034）。
- 说明“目标统计特征 + 稳健分位数候选”方向有效，后续可继续沿该方向做小步迭代。
