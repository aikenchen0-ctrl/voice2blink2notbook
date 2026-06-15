from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    tone_hz: float = 18500.0
    chunk_size: int = 1024
    output_amplitude: float = 0.02
    input_device: Optional[int] = None
    output_device: Optional[int] = None


@dataclass
class DetectorConfig:
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 8.0
    min_energy: float = 0.015
    refractory_s: float = 0.9
    baseline_freeze_s: float = 1.0
    detection_hold_s: float = 2.0
    release_ratio: float = 0.4
    startup_ignore_s: float = 0.0
    warmup_outlier_protection: bool = True


@dataclass
class BlinkConfig:
    method: str = "twinkle"
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 2.5
    min_score: float = 0.006
    refractory_s: float = 1.05
    baseline_freeze_s: float = 0.60
    short_window: int = 9
    phase_pair_lag: int = 3
    phase_step_floor: float = 0.015
    absolute_score_floor: float = 0.0
    startup_ignore_s: float = 2.0
    release_ratio: float = 0.4
    twinkle_peak_gate_enabled: bool = True
    twinkle_peak_min_ratio: float = 1.0
    twinkle_min_peak_score: float = 0.035
    twinkle_max_peak_score: float = 1.2
    twinkle_min_motion_energy: float = 0.0
    twinkle_max_motion_energy: float = 0.22
    twinkle_min_sign_changes: int = 0
    twinkle_max_sign_changes: int = 2
    twinkle_large_motion_score: float = 1.5
    twinkle_large_motion_energy: float = 0.18
    twinkle_large_motion_suppress_s: float = 0.75


@dataclass
class FmcwConfig:
    # FMCW 发射参数：对应 TwinkleTwinkle 论文里的 18-22 kHz、512 点周期、480 点 chirp、32 点 guard。
    f0_hz: float = 18000.0
    f1_hz: float = 22000.0
    sound_speed_m_s: float = 343.0
    period_samples: int = 512
    chirp_samples: int = 480
    guard_samples: int = 32
    tukey_alpha: float = 0.25
    decimation_factor: int = 16
    # 论文 6.4.4 提到接收信号先高通以移除 audible noise；截止频率未给出，这里默认放在 18 kHz chirp 下沿之前。
    raw_highpass_enabled: bool = True
    raw_highpass_hz: float = 17000.0
    raw_highpass_taps: int = 65
    # 接收端混频后先做低通再降采样；每个 decimation phase 的 FIR tap 数越大，抗混叠越强但计算越多。
    decimation_filter_taps_per_phase: int = 8
    valid_start: int = 16
    # 完整 512 点周期会抽取出 32 个 phase point；候选轨迹只用 active chirp 内的 16..29。
    # 论文说只比较 index larger than 15 的点；这里默认 target 用 16..29，reference 用前一个点 15。
    valid_stop: int = 30
    reference_offset: int = 1
    track_count: int = 5
    track_pairs: tuple[tuple[int, int], ...] = ()
    # 实时诊断用固定轨迹：空值时沿用第一条 realtime track；离线 ranking 后可指定最相关 pair。
    fixed_trajectory_pair: tuple[int, int] = ()
    # Eq. 6-8 中 phase difference 与 (N2-N1) 成正比；实时展示轨按 phase-point gap 归一化，便于多轨比较。
    track_gap_normalization: bool = True
    phase_window_size: int = 180
    candidate_interval_length: int = 5
    candidate_intervals_per_criterion: int = 3
    # 论文 §3.2 只说明 trajectory 做归一化和小窗口 moving average；不包含去趋势。
    trajectory_smoothing_window: int = 3
    trajectory_detrend_window: int = 1
    # 论文 4.3.1 用相邻 edge 起点的 0.5s 间隔来分组；实现时按采样率/period 换算成 chirp 行数。
    segmentation_group_gap_s: float = 0.5
    segmentation_sample_rate_hz: int = 48_000
    # 连续实时投票不可能每个 chirp 都跑完整 45 轨迹解码；这里控制刷新频率和最低运动门槛。
    vote_update_periods: int = 20
    vote_min_delta_rms: float = 0.10
    # candidate pending 窗口内的 45 轨迹重算单独节流；最终判定时仍会强制刷新一次。
    confirm_window_vote_update_periods: int = 20
    # phase points 对离线分析有用，但实时 CSV 行会明显变宽；卡顿排查时可临时关闭。
    log_phase_points: bool = True
    # 第一层：candidate 只负责高召回；默认用单轨 delta，更接近旧版单分数 detector。
    candidate_history_size: int = 120
    candidate_min_history: int = 20
    candidate_threshold_k: float = 3.0
    candidate_min_score: float = 0.05
    candidate_refractory_s: float = 1.05
    candidate_baseline_freeze_s: float = 0.75
    candidate_release_ratio: float = 0.4
    candidate_startup_ignore_s: float = 2.0
    candidate_score_source: str = "track0_delta"
    # 第二层：confirm 默认先恢复旧版召回；FMCW vote 先做辅助解释，数据足够后可打开强约束。
    confirm_window_s: float = 0.8
    confirm_min_delta_rms: float = 0.05
    # 最新 blink-only 标注里，primary peak + max_delta_rms<=0.06 可保住 37/39 命中并少 4 个 FP。
    # 这仍是候选确认阈值，不是最终 95% 结论；后续要用含 w 标注的数据继续校准。
    confirm_large_motion_delta_rms: float = 0.06
    confirm_large_motion_duration_s: float = 0.25
    confirm_high_delta_rms: float = 0.10
    confirm_release_ratio: float = 0.4
    confirm_require_vote: bool = False
    confirm_vote_min_confidence: float = 0.20
    confirm_vote_min_stability: float = 0.40
    confirm_single_blink_patterns: tuple[str, ...] = ("1", "11", "12", "21")
    vote_candidate_enabled: bool = False
    vote_candidate_min_confidence: float = 0.60
    vote_candidate_refractory_s: float = 1.05
    # 混合模式：FMCW 负责多轨可视化/二次信息，旧版单频 blink detector 负责主眨眼触发。
    primary_blink_enabled: bool = True
    primary_blink_tone_ratio: float = 1.0
    primary_blink_chirp_ratio: float = 1.0
    primary_blink_peak_enabled: bool = True
    primary_blink_peak_min_score: float = 0.04
    primary_blink_peak_min_ratio: float = 0.8
    # 单频 blink_score 过大通常来自强扰动；旧版 twinkle gate 也有 max_peak_score 上限。
    primary_blink_peak_max_score: float = 0.25
    primary_blink_peak_refractory_s: float = 1.2
    # 混合模式下，18.5 kHz 单频只服务旧版 blink 候选；提取 FMCW phase matrix 前默认把它从录音中投影移除。
    primary_blink_tone_cleanup_enabled: bool = True
    primary_blink_tone_hz: float = 18500.0
    # macOS 输入/输出音频存在系统延迟；FMCW 取相位前需要先把录音周期对齐到 chirp 周期。
    sync_enabled: bool = True
    sync_warmup_blocks: int = 8
    sync_min_confidence: float = 0.12


@dataclass
class CameraConfig:
    enabled: bool = True
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0


@dataclass
class VisualBlinkConfig:
    enabled: bool = True
    auto_mark_blinks: bool = True
    threshold: float = 0.22
    refractory_s: float = 0.25
    model_path: str = "assets/face_landmarker.task"
    max_fps: float = 10.0
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    fmcw: FmcwConfig = field(default_factory=FmcwConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    visual_blink: VisualBlinkConfig = field(default_factory=VisualBlinkConfig)
    mode: str = "wave"
    session_root: str = "sessions"
    window_name: str = "HP Acoustic Hand Wave"
    max_duration_s: Optional[float] = None
    headless: bool = False
    record_video: bool = True
    profile_performance: bool = False
    ui_fps: float = 10.0
    ui_text_fps: float = 5.0
    collection_target_blinks: int = 0
    collection_target_negatives: int = 0

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "audio": asdict(self.audio),
            "detector": asdict(self.detector),
            "blink": asdict(self.blink),
            "fmcw": asdict(self.fmcw),
            "camera": asdict(self.camera),
            "visual_blink": asdict(self.visual_blink),
            "mode": self.mode,
            "session_root": self.session_root,
            "window_name": self.window_name,
            "max_duration_s": self.max_duration_s,
            "headless": self.headless,
            "record_video": self.record_video,
            "profile_performance": self.profile_performance,
            "ui_fps": self.ui_fps,
            "ui_text_fps": self.ui_text_fps,
            "collection_target_blinks": self.collection_target_blinks,
            "collection_target_negatives": self.collection_target_negatives,
        }
