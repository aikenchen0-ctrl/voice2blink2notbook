from pathlib import Path

import numpy as np

from scripts.rank_fmcw_phase_pairs_many import _SessionInput, _per_session_template_summary_rows
from hp_acoustic_wave.fmcw_pair_ranking import PhasePairRank


def test_per_session_template_summary_rows_uses_session_best_pair():
    times = np.arange(120, dtype=np.float64) * 0.05
    phase = np.zeros((times.size, 8), dtype=np.float64)
    phase[:, 5] += np.exp(-0.5 * np.square((times - 3.0) / 0.18))
    phase[:, 5] += np.exp(-0.5 * np.square((times - 5.0) / 0.18)) * 2.0
    session_input = _SessionInput(
        session_dir=Path("sessions/example"),
        times=times,
        phase_matrix=phase,
        phase_source="test",
        blink_markers=(3.0,),
        negative_markers=(5.0,),
        valid_start=5,
        valid_stop=7,
    )
    rank = PhasePairRank(
        reference_index=4,
        target_index=5,
        blink_peak_median=1.0,
        background_peak_p95=0.1,
        negative_peak_p95=0.2,
        decision_threshold=0.2,
        separation=2.0,
        blink_hit_rate=1.0,
        background_trigger_rate=0.0,
        negative_trigger_rate=0.0,
        blink_peak_count=1,
        background_window_count=2,
        negative_window_count=1,
    )

    rows = _per_session_template_summary_rows(
        [(session_input, (rank,))],
        center_offset_s=0.0,
        marker_window_s=0.4,
        background_window_s=0.4,
        template_points=9,
    )

    groups = {row["group"] for row in rows}
    assert groups >= {"blink", "background", "negative"}
    assert {row["best_pair"] for row in rows} == {"4:5"}
    assert all(row["session"] == "example" for row in rows)
