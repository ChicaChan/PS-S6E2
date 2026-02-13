# PS-S6E2 项目进度记录

> 最后更新：2026-02-13 19:10:00 +08:00

## 1. 目标与当前状态

- 目标：冲刺 Kaggle `playground-series-s6e2` 前 10%
- 当前榜单规模：`2387` 条 public 排名
- 前 20% 阈值：`Rank 477`，分数 `0.95367`
- 前 10% 阈值：`Rank 238`，分数 `0.95386`
- 当前队伍：`Chica Chan`
- 当前成绩：`Rank 261`，`Public Score 0.95385`
- 与前 10% 分差：约 `+0.00001`

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


## 8. v14 执行记录（Top10% 冲刺框架落地）

### 8.1 代码侧升级
- 云端训练脚本新增参数与能力：
  - `target_stats_interactions_grid`：支持一次运行多组 target-stats 交叉配置
  - `target_stats_bin_count`：为高基数数值列生成分箱统计特征
  - `quantile_grid`：批量生成 `robust_q*` 分位数候选
  - `candidate_max_corr`：候选挑选阶段相关性约束
  - `anchor_file + anchor_weight_grid`：自动输出锚定混合候选
- `online_sprint.py` 新增策略化日常挑选：
  - `strategy`: `meta_first/robust_first/anchor_safe`
  - `max_submit_count`、`anchor_path`、`anchor_weight`
- `blend_submissions.py` 新增：
  - `method=quantile`
  - `corr_prune_threshold`（融合前相关性剪枝）

### 8.2 本地验证
- 全量单测：`40 passed`
- 本地 smoke（小样本）已跑通：
  - 多交叉网格：`ts_i4, ts_i6`
  - 分位数候选：`robust_q0p25`, `robust_q0p5`
  - 产物与指标文件生成正常

### 8.3 云端训练与提交（2026-02-12）
- Kernel：`v14`
- 云端最优候选：`aggressive_ts_i4`（CV 约 `0.9555903`）
- 三次计分提交：
  1. `submission_best.csv` -> `0.95374`
  2. `submission_tuned_lgb_cat_ts_i4.csv` -> `0.95374`
  3. `submission_anchor_v13robust_v14best_30_70.csv` -> `0.95374`

### 8.4 结论
- v14 冲刺框架已完整落地并可持续迭代，但本轮 Public 仍持平 `0.95374`。
- 距离当前 Top10% 门槛（约 `0.95386`）仍差约 `0.00012`，下一轮应聚焦更高区分度候选（优先扩展 quantile 网格与更低相关性组合）。

## 9. v15 云端执行记录（严格云训练）

### 9.1 执行约束
- 已按要求停止本地训练进程；本轮仅使用 Kaggle 云端 Kernel 训练。
- 本地仅保留提交文件校验与结果分析，不进行任何训练。

### 9.2 云端训练
- Kernel：`chicachan/ps-s6e2-optimized-hybrid-gpu`
- 版本：`v15`
- 主要配置升级：
  - `target_stats_interactions_grid=3,4,6`
  - `target_stats_bin_count=12`
  - `quantile_grid=0.15,0.20,0.25,0.30,0.35,0.40,0.50`
  - `candidate_max_corr=0.9985`
- 云端最优候选：`aggressive_ts_i4`（CV `0.9555910`）

### 9.3 本轮提交结果（2026-02-12 UTC）
1. `submission_robust_q0p5_ts_i4.csv` -> `0.95373`
2. `submission_aggressive_ts_i4.csv` -> `0.95374`

### 9.4 榜单快照（2026-02-12 16:38:52 UTC）
- Public 榜总队伍：`2285`
- Top 20% 阈值：`Rank 457`，`Score 0.95367`
- Top 10% 阈值：`Rank 228`，`Score 0.95386`
- 当前队伍 `Chica Chan`：`Rank 332`，`Score 0.95374`
- 距离 Top 10% 仍差：约 `0.00012`

### 9.5 结论
- v15 在云端完整跑通并完成提交，但最佳 Public 仍为 `0.95374`（持平历史最好）。
- 下一轮应继续围绕“低相关高区分候选 + 锚定稳健混合”做小步试探，避免同质化提交。

## 10. v16 云端执行记录（跨 profile 稳健候选）

### 10.1 代码升级
- 云端脚本新增跨 profile 候选构建：
  - `global_quantile_grid`：对全局稳健池生成 `global_robust_q*` 候选
  - `global_corr_prune_threshold`：全局分位候选专用相关性剪枝阈值
- 新增 `build_robust_pool`，统一复用稳健基候选筛选逻辑，避免重复代码。

### 10.2 云端训练
- Kernel：`chicachan/ps-s6e2-optimized-hybrid-gpu`
- 版本：`v16`
- 训练状态：`KernelWorkerStatus.COMPLETE`
- 本轮最佳候选：`aggressive_ts_i4`（CV `0.9555903`）
- 新增全局候选示例：`global_robust_q0p18/q0p25/q0p33/q0p4/q0p5`

### 10.3 提交执行状态（代理 7890）
- 已完成云端训练产物下载与本地 submission 校验。
- 通过本地代理 `127.0.0.1:7890` 重试提交成功，三次计分提交结果如下（2026-02-13 UTC）：
  1. `submission_aggressive_ts_i4.csv` -> `0.95374`
  2. `submission_tuned_lgb_cat_ts_i4.csv` -> `0.95374`
  3. `submission_global_robust_q0p5.csv` -> `0.95371`

### 10.4 结论
- v16 新增全局稳健候选已完成线上验证，但本轮最优 Public 仍为 `0.95374`（持平历史最好）。
- `global_robust_q0p5` 在线分数低于主候选，后续可保留该思路但需收缩全局池范围再试。


### 10.5 榜单快照（2026-02-13 09:24:18 UTC）
- Public 榜总队伍：`2372`
- Top 20% 阈值：`Rank 474`，`Score 0.95367`
- Top 10% 阈值：`Rank 237`，`Score 0.95386`
- 当前队伍 `Chica Chan`：`Rank 348`，`Score 0.95374`
- 距离 Top 10% 仍差：约 `0.00012`

## 11. v17 云端执行记录（外部高分融合）

### 11.1 参考与落地
- 参考高分公开方案并提取可复用策略：
  - `achilov15/corrprune-median-blend-0-95397`（相关性剪枝 + 稳健中位/分位融合）
  - `omidbaghchehsaraei/the-best-solo-model-so-far-realmlp-lb-0-95397`（高质量单模型输出）
  - `harukiharada/realmlp-ext-target-stats-5-fold-cv`（target-stats + neural 侧输出）
  - `anthonytherrien/predicting-heart-disease-blend`（公开融合输出）
- 本仓库代码改造：
  - `blend_submissions.py` 新增来源分组与配额约束：`input_groups`、`min_per_group`
  - 新增 `quantile_grid` 一次产出多份 submission（`q25/q33/q50`）
  - `online_sprint.py` 新增 `external_mix_first` 策略，并输出 `source_mix/corr_to_internal_best/expected_role`
  - 云端脚本新增 `allow_global_best` 与 `selection_pool/internal_best_set` 指标，避免全局候选误抢 best

### 11.2 云端训练
- Kernel：`chicachan/ps-s6e2-optimized-hybrid-gpu`
- 版本：`v17`
- 训练状态：`KernelWorkerStatus.COMPLETE`
- 云端最优内部候选：`aggressive_ts_i4`（CV `0.9555907`）

### 11.3 外部融合候选生成
- 外部候选目录：`kaggle_outputs/external_candidates/v17`
- 生成融合文件：
  - `submission_external_mix_q0p25.csv`
  - `submission_external_mix_q0p33.csv`
  - `submission_external_mix_q0p5.csv`
- 融合策略：`corr_prune_threshold=0.9988` + `min_per_group={internal:1, external:1}`

### 11.4 本轮提交结果（2026-02-13 UTC）
1. `submission_aggressive_ts_i4.csv` -> `0.95374`
2. `submission_external_mix_q0p5.csv` -> `0.95376`
3. `submission_external_mix_q0p25.csv` -> `0.95385`

### 11.5 榜单快照（2026-02-13 10:36:47 UTC）
- Public 榜总队伍：`2387`
- Top 20% 阈值：`Rank 477`，`Score 0.95367`
- Top 10% 阈值：`Rank 238`，`Score 0.95386`
- 当前队伍 `Chica Chan`：`Rank 261`，`Score 0.95385`
- 距离 Top 10% 仍差：约 `0.00001`

### 11.6 结论
- v17 外部高分融合方向显著有效：从 `0.95374` 抬升到 `0.95385`。
- 已逼近 Top10% 门槛，下一轮优先围绕 `external_mix_q0p25` 做极小扰动（q 或 corr 阈值微调）冲击 `0.95386+`。

## 12. 本地仓库状态（代码已就绪）

- 本轮已完成代码实现、云端训练、线上提交与记录回填。
- 当前变更覆盖模块：
  - `kaggle_kernel/ps_s6e2_gpu/train_gpu_catboost.py`
  - `src/blend_submissions.py`
  - `src/online_sprint.py`
  - `tests/test_blend_submissions.py`
  - `tests/test_online_sprint.py`
  - `tests/test_train_gpu_catboost_cloud.py`
  - `PROJECT_PROGRESS.md`
- 当前测试状态：`python -m pytest -q` 全量通过（47 passed）。

