# TODO

日期：2026-06-13

## 当前主线

目标不是让 FMCW 立刻替代旧版 `blink`，而是把旧版稳定的短窗口候选触发与 FMCW 多轨二次确认组合起来。

```text
短窗口 candidate_score
-> rolling median/MAD candidate_threshold
-> candidate event
-> 候选附近 FMCW 多轨窗口
-> confirm / suppress
```

## 2026-06-15：论文对齐主线

已完成：

- [x] 全文重新核对 TwinkleTwinkle 的 FMCW 链路。
- [x] 写出 `论文对齐.md`，明确论文步骤、当前代码位置、缺口和下一步。
- [x] 修正 `FmcwConfig.valid_start/valid_stop`：完整 phase matrix 保存 32 点，但候选 target 从 16 开始、stop 到 active chirp decimation 末尾，保留论文 “index larger than 15” 的有效相位点。
- [x] `FmcwFeature` 新增完整 `phase_points`。
- [x] `features.csv` / `manual_markers.csv` 新增 `fmcw_phase_points`。
- [x] 离线分析新增 `paper_style_vote_from_rows`，可从新 session 的 phase matrix 重算 45 条 trajectory vote。
- [x] 单元测试覆盖 phase point 写出和 paper-style vote 摘要。
- [x] 抽出 `decode_phase_matrix_vote`，让离线分析和实时 confirm 共用同一套 45 轨迹投票逻辑。
- [x] 实时 confirm 新增候选窗口 `fmcw_confirm_window_*` 指标。
- [x] UI 显示候选窗口 `win=` 和 `conf=`。
- [x] CSV/marker 写入候选窗口重算投票字段。
- [x] 补齐论文 4.3.3：异常方向序列用 window=4、step=2 的方差曲线，并按局部 high-low-high motif 计数。
- [x] 对齐论文 4.2：候选选择内部区分 chirp 内 unwrap 与跨 chirp temporal unwrap。
- [x] 对齐论文 4.2：三准则重复 interval 时切换 `t2-1/t2-2/t2-3` reference，保留不同 phase-pair 视角。
- [x] 对齐论文 4.2：替代 reference 加下界，不再回退到 `valid_start - 1` 之前的低通边界风险点。
- [x] 对齐论文 4.2：替代 reference 也不可用时，显式退到同 criterion 的下一个 unused interval，并补测试记录这是工程退路。
- [x] 对齐论文 4.3.2：fine selection 后重新按时间间隔分组，不再把远距离剩余边硬拼进同一 segment。
- [x] 对齐论文 4.3.2：two-edge segment 中任一 edge diff 达到 `0.4 * global max_diff` 时保留两条边，补测试。
- [x] 对齐论文 4.3.4：over-pruning correction 只补 refined segment 范围内的中间弱边。
- [x] 对齐论文第 5 节：三组各 15 条先取 winner，再按组内 winner 票数选最终 rhythm。
- [x] 对齐论文第 5 节：投票三组改为按 `criterion` 显式分组，不再隐式依赖 bundle 顺序三等分。
- [x] 对齐论文 6.4.3：命令行未显式传 `--fmcw-valid-stop` 时，自动用 `chirp_samples // decimation_factor`。
- [x] 对齐论文 3.2：接收端按完整 512 点周期混频、windowed-sinc FIR 低通、decimation，得到 32 个 phase points。
- [x] CLI 新增 `--fmcw-decimation-filter-taps-per-phase`，用于调节实时低通抽取强度。
- [x] 对齐论文 Eq. 6-8：实时 `track_0..track_4` 默认按 phase-point gap 归一化，避免宽间隔 track 天然放大。
- [x] CLI 新增 `--fmcw-no-track-gap-normalization`，可恢复旧可视化尺度做 A/B。
- [x] 混合实时模式默认在 FMCW 解调前移除 18.5 kHz 单频，降低旧版候选音对 phase matrix 的污染。
- [x] CLI 新增 `--fmcw-no-primary-blink-tone-cleanup`，可保留混合原始输入做 A/B。
- [x] 离线分析新增 `marker_trajectory_details.csv`，展开每个 marker 窗口的 45 条论文候选轨迹。
- [x] 对齐论文 4.3.3：方向序列解码收紧，5 个方向这类异常形态不再直接猜成 2 次眨眼，而交给局部 high-low-high variance fallback。
- [x] 对齐论文 6.4.4：FMCW 相位提取前新增 raw high-pass，默认 17 kHz，用于压制可闻背景噪声。
- [x] CLI 新增 `--fmcw-no-raw-highpass` / `--fmcw-raw-highpass-hz` / `--fmcw-raw-highpass-taps` 做 A/B。
- [x] 对齐论文 3.1：FMCW 发射 chirp 从 `sin(phi_t)` 改为论文定义的 `cos(phi_t)`。
- [x] 对齐论文 4.3.1：edge trim 起点修正到“阈值差值后的样本”，不再少剪 1 个点。
- [x] 对齐论文 4.3.1：edge trim 触发条件改为严格大于 `1/4 * max_diff`，匹配 larger than 表述。
- [x] 补齐 macOS 实时 FMCW 录音周期同步：估计 `fmcw_sync_lag_samples` 后再切 chirp 帧。
- [x] CSV/marker 新增 `fmcw_sync_lag_samples`、`fmcw_sync_confidence`，用于判断 phase matrix 是否先同步成功。
- [x] CLI 新增 `--fmcw-no-sync` / `--fmcw-sync-warmup-blocks` / `--fmcw-sync-min-confidence` 做 A/B。
- [x] 对齐论文 4.2：45 轨迹候选改为三组独立评分：轨迹相似、chirp 内 slope 稳定、跨 chirp 同点时间稳定。
- [x] 对齐论文 4.2：跨 chirp 同点时间稳定分数改为相邻 chirp 绝对差值总和，避免线性漂移被误判稳定。
- [x] 对齐论文 4.3.4：over-pruning correction 补边候选从“完全包含 refined 范围”修正为“与 refined 范围重叠”。
- [x] 对齐论文 3.2：`phase_difference_trajectory` 改为 chirp 内先 unwrap，再做 phase subtraction，最后对时间轨迹 unwrap。
- [x] 对齐论文 3.2：FMCW 论文式投票默认关闭 trajectory detrend，仅保留 normalize + 小窗口 moving average。
- [x] `smooth_trajectory` / `detrend_trajectory` 改为 edge padding，避免零填充在窗口两端制造假波动。
- [x] CLI 新增 `--fmcw-trajectory-detrend-window` / `--fmcw-trajectory-smoothing-window` 做 A/B。
- [x] 离线分析读取 session `metadata.json` 里的 FMCW 参数，避免历史 session 被新默认重解释。
- [x] 修正实时 confirm 决策顺序：强 45 轨迹窗口 vote 支持 blink 时，先确认眨眼，再考虑大动作幅度门控。
- [x] 对齐论文 4.3.1：edge 分组间隔改为按 `0.5s` 换算 chirp 行数，不再固定写死 `47`。
- [x] CLI 新增 `--fmcw-segmentation-group-gap`，用于 A/B 调整论文的 0.5s 分组阈值。
- [x] 离线 `confirm_features.csv` 同步实时 confirm 顺序：paper/window vote 支持 blink 时，不再先被 large-motion 阈值压制。
- [x] 对齐论文 4.3.1：coarse edge 候选按“more than two sampling points”保留 3 点线段，不再把 `end-start=2` 的边过早丢弃。
- [x] 对齐论文 4.3.1：edge 分组用 “less than 0.5s” 的严格不等式；默认 `512/48k` 最大间隔为 46 行而不是四舍五入的 47 行。
- [x] 对齐论文 4.3.1：flat peak/valley 的 extrema 取平台中点，避免平台左端切边扭曲 edge 长度。
- [x] 对齐论文 4.3.1：coarse edge length 阈值按方向独立取 longest edge，补测试防止误改成全局最长边。
- [x] 对齐论文 4.3.1：representative difference 用 top-3 adjacent diff 平均值，补测试防止误改成 max/全均值。

下一步：

- [ ] 新一轮采集必须同时标 `b` 和 `w`，并等人类明确说“好了”后再评估。
- [x] 新增跨 session 聚合评估脚本 `scripts/evaluate_candidate_fusion_many.py`。
- [x] 将主候选默认调为 `min_score=0.04`、`min_ratio=0.80`、`refractory=1.20`。
- [x] 在 `hp_fmcw_20260615_171453 + hp_fmcw_20260615_180004` 上验证默认聚合 `fused recall=0.958`。
- [ ] 对最新带 `fmcw_blink_vote_evidence` / `fmcw_blink_trajectory_value` 的 session 跑 `scripts/evaluate_candidate_fusion.py`。
- [ ] 对新 b/w session 跑 `scripts/evaluate_candidate_fusion_many.py`，确认默认参数在新数据上仍有 `recall >= 0.95`。
- [ ] 新 b/w session 里优先检查 `fmcw_fixed_trajectory_distance_mm`：眨眼附近是否有稳定峰，w 附近是否形态不同。
- [ ] 从 `fused_candidate_diagnostics.csv` 检查 fallback TP/FP 的 `dominant_fmcw_blink_trajectory_pair`、`trajectory_value`、`vote_evidence` 差异。
- [ ] 从 `fused_candidate_diagnostics.csv` 检查 fallback TP/FP 的 `max_abs_fmcw_fixed_trajectory_distance_mm` 差异。
- [ ] 从 `candidate_fusion_fmcw_sweep.csv` 选择能保持 blink recall >= 0.95 且明显降低 FP 的阈值组合。
- [ ] 同一张 sweep 表必须同时检查 `false_positive` 和 `negative_conflict_total`，不能只看 recall。
- [x] 新增 `fmcw_pair_ranking.py`，离线统计 phase-pair 与 blink/w/background 的分离度。
- [x] 新增 `scripts/rank_fmcw_phase_pairs_many.py`，跨 session 聚合寻找 FMCW 最相关物理轨迹线。
- [x] 修正 `scripts/rank_fmcw_phase_pairs.py`，target 不再落到 `valid_start` 之前的无效区间。
- [x] 新增 `--fmcw-fixed-trajectory-pair`，允许把离线选出的 pair 接到实时洋红固定 mm 轨迹。
- [x] 将 phase-pair ranking 从 delta peak 扩展到窗口轨迹形态特征，包括 span、峰谷方向、回落程度。
- [ ] 只有 `recommended_pair` 非空时，才把该 pair 用 `--fmcw-fixed-trajectory-pair` 接入页面实测。
- [x] 新增 center_offset sweep，检查人工 b 标记延迟是否导致 shape window 没截中真实眨眼。
- [x] 为 ranking top pair 输出 blink/background/large_motion 平均轨迹模板，直观看形态差异。
- [x] 新增模板诊断 summary，量化 peak_relative_time、span、endpoint_delta、sign、blink_vs_background_template_distance。
- [x] 检查不同 session 的 blink 模板是否方向相反；如相反，ranking 需要做 polarity alignment。
- [x] 新增 `scripts/evaluate_fixed_posture_study.py`，统一跑标注数量、phase-pair ranking、template diagnostics、candidate fusion 和总决策。
- [x] 新增 `scripts/run_fixed_posture_collection.py`，固定姿态采集默认保留 `fmcw_phase_points`。
- [ ] 重新采集一组固定姿态 b/w 数据，至少 40 blink + 20 w，并保留 `fmcw_phase_points`。
- [ ] 对固定姿态新数据运行 `scripts/evaluate_fixed_posture_study.py`。
- [ ] 若固定姿态数据仍无 `recommended_pair`，将 FMCW 定位降级为大动作压制/诊断，不再追求单独物理眨眼线。
- [ ] 若 `recommended_pair` 持续为空，优先排查 chirp 同步、phase point 有效区间、轨迹评分口径和人脸几何。
- [ ] 若新 session 没有 `w` 标注，只能评估眨眼召回，不能宣称抗干扰达标。
- [ ] 采一轮新 session，先看 `fmcw_sync_confidence` 是否高于阈值、`fmcw_sync_lag_samples` 是否非空。
- [ ] 新 session 默认应显示 `trajectory_detrend_window=1`；若要对比旧工程处理，启动时加 `--fmcw-trajectory-detrend-window 31`。
- [ ] 观察实时 UI 中 `win=` / `conf=` 是否在 blink 后稳定出现。
- [ ] 如需严格论文链路采集，启动时加 `--fmcw-no-primary-blink`；默认混合模式虽会清理 18.5k 单频，但仍不是纯论文发射。
- [ ] 运行 `scripts/analyze_fmcw_session.py`，查看 `confirm_features.csv` 中的 `paper_vote_*` 字段。
- [ ] 同时查看 `marker_trajectory_details.csv`，定位误报来自哪个 criterion / phase-pair / trajectory。
- [ ] 对 blink / large_motion 分别统计 `paper_vote_pattern`、`paper_vote_confidence`、`paper_vote_score`。
- [ ] 对比低通抽取前后的新 session：重点看 `fmcw_phase_points` 是否更平滑、`paper_vote_*` 是否更稳定。
- [ ] 对比 raw high-pass 开/关的新 session：重点看可闻噪声场景下 `fmcw_phase_points` 和 `paper_vote_*` 是否更稳定。
- [ ] 注意新 session 的 `fmcw_track_*` 已默认做 gap 归一化；若要对比旧 session 的绝对幅度，需要加 `--fmcw-no-track-gap-normalization`。
- [ ] 对实时 `features.csv` 统计 `fmcw_confirm_window_pattern`、`fmcw_confirm_window_confidence`、`fmcw_confirm_window_candidate_count`。
- [ ] 注意：当前 `paper_vote_confidence` / `fmcw_confirm_window_confidence` 是获胜组内 `score / 15`，不是 45 条全局占比。
- [ ] 若窗口 vote 对 blink 有稳定支持，考虑打开 `--fmcw-confirm-require-vote` 或降低旧版候选误报权重。
- [ ] 若 paper-style vote 对 large_motion 仍稳定误投 blink，导出/查看单条 trajectory 的 segmentation 中间产物，不急着改实时阈值。

## 已完成

- [x] 保留 `wave` / `blink` 旧模式。
- [x] 新增 `--mode fmcw`。
- [x] 实现 FMCW chirp 播放和接收处理。
- [x] 实现 rolling phase matrix。
- [x] 实现 5 条实时 track 可视化。
- [x] 实现 45 条 candidate trajectory、segmentation、voting。
- [x] 实现真实 session 回放分析脚本。
- [x] 生成两轮真实数据分析报告。
- [x] 确认 `fmcw_track_delta_rms` 只能做候选召回，不能做最终 blink 判定。
- [x] 接入 FMCW candidate detector：`candidate_score -> rolling median/MAD -> candidate event`。
- [x] UI 叠加 candidate score 和 threshold 线。
- [x] CSV/marker 写入 `fmcw_candidate_*` 字段。
- [x] 单元测试覆盖 CLI 参数映射、FMCW vote、FMCW candidate detector。
- [x] 扩展 `fmcw_session_analysis.py`，重放 candidate detector。
- [x] 对每个 marker 统计 candidate 命中。
- [x] 输出 candidate 召回率和 large_motion 触发率。
- [x] 对 `candidate_threshold_k` / `candidate_min_score` 做 sweep。
- [x] 把两轮真实 session 都纳入同一张对比表。
- [x] 实现第一版离线 `FMCWConfirmFeature`。

## 当前结论

默认 candidate detector 已调整为召回优先：

```text
candidate_threshold_k = 3.0
candidate_min_score = 0.05
```

两轮真实 session 上：

```text
第一轮 blink: 13/13 candidate 命中
第一轮 large_motion: 6/6 candidate 命中
第二轮 blink: 9/9 candidate 命中
第二轮 large_motion: 5/5 candidate 命中
```

这符合两阶段设计：candidate 负责召回，不负责最终判断。

## 下一个执行单元：调校 FMCWConfirmFeature

验收：

```text
能回答：哪些 candidate 应该 confirmed，哪些应该 suppressed。
```

- [x] 在 marker 附近截取 `t - 0.8s` 到 `t + 0.8s`。
- [x] 计算 `max_delta_rms`。
- [x] 计算 `median_delta_rms`。
- [x] 计算 `high_delta_duration_s`。
- [x] 计算 `max_confidence`。
- [x] 计算 `pattern_rows`。
- [x] 计算 `pattern_stability`。
- [x] 计算 `dominant_pattern`。
- [x] 计算多轨峰宽和同步程度。
- [x] 标记 `large_motion_suppressed`。
- [x] 标记 `fmcw_confirmed`。
- [ ] 对第一轮低幅 large_motion 继续找区分特征。
- [ ] 对 `large_motion_delta_threshold` / `large_motion_duration_threshold_s` 做 sweep。
- [ ] 增加 confirm 参数 sweep 报告。
- [ ] 把 confirm 结果接入实时 UI，但先不作为最终 blink 输出。

## 当前执行单元：实时最终 blink 判定层

必须新增独立输出，不再把 candidate 当最终 blink：

```text
fmcw_candidate
-> realtime confirm window
-> fmcw_confirmed_blink 或 fmcw_suppressed_motion
```

实现任务：

- [x] 保留 `fmcw_candidate` 事件，作为疑似触发。
- [x] 新增 `fmcw_confirmed_blink` 事件，作为最终眨眼判定。
- [x] 新增 `fmcw_suppressed_motion` 事件，记录被压制的大动作/干扰。
- [x] 新增 realtime confirm 滚动窗口。
- [x] 实时计算 confirm 特征：
  - [x] `fmcw_confirm_max_delta_rms`
  - [x] `fmcw_confirm_high_delta_duration_s`
  - [x] `fmcw_confirm_pattern`
  - [x] `fmcw_confirm_confidence`
  - [x] `fmcw_confirm_state`
- [x] CSV 写入 confirm 字段。
- [x] marker 快照写入 confirm 字段。
- [x] UI 同时显示：
  - [x] candidate score/threshold
  - [x] confirmed blink 状态
  - [x] suppressed motion 状态
- [x] 事件生命周期：一次 candidate 只能确认/压制一次，不能持续动作重复刷最终事件。
- [x] 单元测试覆盖：
  - [x] blink-like candidate 产生 `fmcw_confirmed_blink`
  - [x] large-motion-like candidate 产生 `fmcw_suppressed_motion`
  - [x] UI 主标题仍以 blink 为目标。

第一版策略：

```text
max_delta_rms >= 0.20 或 high_delta_duration_s >= 0.25
=> suppressed_motion

0.05 <= max_delta_rms <= 0.20 且 high_delta_duration_s < 0.25
=> confirmed_blink
```

注意：

```text
这是第一版最终判定层，不代表参数已经最终调好。
它的价值是把 candidate 和最终 blink event 在代码、CSV、UI 上分离。
```

验证：

```text
35 passed
sessions_smoke/hp_fmcw_20260613_194930 写出 fmcw_confirm_* 字段
smoke events 中已出现 fmcw_candidate 和 fmcw_suppressed_motion
```

## 2026-06-14：实时连续 vote-based confirm 可视化

已完成：

- [x] confirm history 持续记录实时投票摘要。
- [x] `fmcw_confirmed_blink` 不再只依赖 `dRMS/duration`，还要求 vote support。
- [x] 新增实时字段：
  - [x] `fmcw_confirm_vote_score`
  - [x] `fmcw_confirm_pattern_rows`
  - [x] `fmcw_confirm_pattern_stability`
- [x] UI 显示 `conf` 和 `stab`。
- [x] CLI 支持调参：
  - [x] `--fmcw-confirm-vote-min-confidence`
  - [x] `--fmcw-confirm-vote-min-stability`
- [x] 单元测试覆盖无 vote support 时 `rejected_vote`。

当前实时第一版确认条件：

```text
不是 large motion
max_delta_rms >= confirm_min_delta_rms
dominant pattern in ("1", "11", "12", "21")
max confidence >= confirm_vote_min_confidence
pattern stability >= confirm_vote_min_stability
```

验证：

```text
36 passed
sessions_smoke/hp_fmcw_20260614_133720 写出新字段
```

下一步：

- [ ] 启动页面采一轮 blink / large_motion。
- [ ] 看 `fmcw_confirm_pattern_rows` 是否太少。
- [ ] 如果一直 `rejected_vote`，优先降低 `confirm_vote_min_stability` 或缩短 `vote_update_periods`。
- [ ] 如果 large_motion 被 confirmed，优先提高 `confirm_large_motion_delta_rms` 或加入 track shape 特征。

验收：

```text
blink marker 附近大多数 candidate 被 confirmed。
large_motion 中高幅、长持续、低稳定性的 candidate 被 suppressed。
```

## 暂时不做

- [ ] 不把 `fmcw_pattern` 直接当最终事件。
- [ ] 不继续只凭肉眼调 `vote_min_delta_rms`。
- [ ] 不引入机器学习模型。
- [ ] 不追求完全复现论文离线流程。
- [ ] 不在缺少回放指标时继续堆 UI。

## 2026-06-15：下一轮固定姿态验证

- [x] 增加 FMCW physical-line template-correlation 排名。
- [x] 在 `rank_fmcw_phase_pairs_many.py` 输出 `best_line_pair` / `recommended_line_pair`。
- [x] 在 `evaluate_fixed_posture_study.py` 接入 physical-line gate。
- [x] 用旧三段 session 跑基线：`best_line_pair=11:22`，`hit=0.083`，不推荐。
- [x] 固定姿态采集页面显示 `b=当前/40 w=当前/20` 进度。
- [ ] 启动固定姿态采集，至少标注 40 个 `b` 和 20 个 `w`。
- [ ] 采完后跑 `scripts/evaluate_fixed_posture_study.py`。
- [ ] 若 `recommended_line_pair != null`，把它作为 `--fmcw-fixed-trajectory-pair` 接入实时洋红线复测。
- [ ] 若仍为 `null`，FMCW 降级为干扰压制/诊断，主检测继续走旧版 `blink_score`。
