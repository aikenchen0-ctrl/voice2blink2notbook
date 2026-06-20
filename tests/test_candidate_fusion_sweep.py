import csv

from hp_acoustic_wave.candidate_fusion_sweep import (
    best_sweep_rows,
    sweep_candidate_fusion_session,
    write_candidate_fusion_sweep_outputs,
)


def test_sweep_candidate_fusion_prefers_cleaner_high_score_primary(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    feature_rows = [
        _feature_row(0.0, 0.01, 0),
        _feature_row(1.0, 0.12, 0),
        _feature_row(1.1, 0.01, 0),
        _feature_row(3.0, 0.05, 0),
        _feature_row(3.1, 0.01, 0),
    ]
    marker_rows = [{"time_s": "1.0", "label": "blink", "key": "b"}]
    _write_csv(session / "features.csv", feature_rows)
    _write_csv(session / "manual_markers.csv", marker_rows)

    result = sweep_candidate_fusion_session(
        session,
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        primary_min_scores=(0.04, 0.10),
        primary_min_ratios=(0.8,),
        primary_max_scores=(0.25,),
        primary_refractory_values=(0.5,),
        fallback_enabled_values=(False,),
    )

    assert len(result.rows) == 2
    clean = best_sweep_rows(result.rows, min_recall=1.0, limit=1)[0]
    assert clean.primary_min_score == 0.10
    assert clean.true_positive == 1
    assert clean.false_positive == 0
    assert clean.precision == 1.0


def test_sweep_candidate_fusion_can_write_csv(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_csv(
        session / "features.csv",
        [
            _feature_row(0.0, 0.01, 0),
            _feature_row(1.0, 0.12, 0),
            _feature_row(1.1, 0.01, 0),
        ],
    )
    _write_csv(session / "manual_markers.csv", [{"time_s": "1.0", "label": "blink", "key": "b"}])

    result = sweep_candidate_fusion_session(
        session,
        tolerance_s=0.3,
        ignore_startup_s=0.0,
        primary_min_scores=(0.10,),
        primary_min_ratios=(0.8,),
        primary_max_scores=(0.25,),
        primary_refractory_values=(0.5,),
        fallback_enabled_values=(False,),
    )
    write_candidate_fusion_sweep_outputs(result, tmp_path / "out")

    assert (tmp_path / "out" / "candidate_fusion_parameter_sweep.csv").exists()


def _feature_row(time_s, blink_score, twinkle_peak):
    return {
        "time_s": str(time_s),
        "blink_score": str(blink_score),
        "blink_threshold": "0.05",
        "blink_method": "twinkle",
        "twinkle_candidate_peak": str(twinkle_peak),
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
