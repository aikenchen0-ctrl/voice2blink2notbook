import csv

from hp_acoustic_wave.visual_label_audit import audit_visual_labels


def test_visual_label_audit_reads_visual_features_and_compares_marker_count(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_csv(
        session / "visual_features.csv",
        [
            {
                "time_s": "1.0",
                "timestamp_ms": "1000",
                "available": "1",
                "face_found": "1",
                "left_ear": "0.10",
                "right_ear": "0.11",
                "left_closed": "1",
                "right_closed": "1",
                "is_blink_event": "1",
                "blink_count": "1",
                "inference_ms": "8.0",
                "error": "",
            },
            {
                "time_s": "2.0",
                "timestamp_ms": "2000",
                "available": "1",
                "face_found": "1",
                "left_ear": "0.30",
                "right_ear": "0.11",
                "left_closed": "0",
                "right_closed": "1",
                "is_blink_event": "1",
                "blink_count": "2",
                "inference_ms": "8.0",
                "error": "",
            },
            {
                "time_s": "3.0",
                "timestamp_ms": "3000",
                "available": "1",
                "face_found": "0",
                "left_ear": "0.00",
                "right_ear": "0.00",
                "left_closed": "0",
                "right_closed": "0",
                "is_blink_event": "0",
                "blink_count": "2",
                "inference_ms": "8.0",
                "error": "",
            },
        ],
    )
    _write_csv(
        session / "manual_markers.csv",
        [
            {"time_s": "1.0", "label": "blink", "key": "v"},
        ],
    )

    audit = audit_visual_labels(session)

    assert audit.summary.source == "visual_features"
    assert audit.summary.row_total == 3
    assert audit.summary.visual_event_total == 2
    assert audit.summary.valid_visual_event_total == 1
    assert audit.summary.invalid_visual_event_total == 1
    assert audit.summary.visual_marker_total == 1
    assert audit.summary.visual_event_marker_mismatch == 1
    assert audit.rows[1].issue == "event_not_both_closed"


def test_visual_label_audit_falls_back_to_visual_marker_snapshots(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    _write_csv(
        session / "manual_markers.csv",
        [
            {
                "time_s": "1.0",
                "label": "blink",
                "key": "v",
                "visual_available": "1",
                "visual_face_found": "1",
                "visual_left_ear": "0.10",
                "visual_right_ear": "0.11",
                "visual_left_closed": "1",
                "visual_right_closed": "1",
                "visual_is_blink_event": "1",
                "visual_blink_count": "1",
                "visual_inference_ms": "9.0",
                "visual_error": "",
            },
            {
                "time_s": "2.0",
                "label": "blink",
                "key": "b",
                "visual_available": "1",
                "visual_face_found": "1",
            },
        ],
    )

    audit = audit_visual_labels(session)

    assert audit.summary.source == "manual_markers"
    assert audit.summary.row_total == 1
    assert audit.summary.visual_event_total == 1
    assert audit.summary.valid_visual_event_total == 1
    assert audit.summary.visual_marker_total == 1
    assert audit.rows[0].valid_event


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
