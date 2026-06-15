from hp_acoustic_wave.fmcw_template_diagnostics import summarize_template_rows


def test_summarize_template_rows_reports_shape_and_blink_distance():
    rows = []
    blink_values = [0.0, 1.0, 0.0]
    background_values = [0.0, 0.1, 0.2]
    for group, values in (("blink", blink_values), ("background", background_values)):
        for index, value in enumerate(values):
            rows.append(
                {
                    "group": group,
                    "reference_index": "24",
                    "target_index": "25",
                    "center_offset_s": "0.300000",
                    "point_index": str(index),
                    "relative_time": str(-1.0 + index),
                    "mean": str(value),
                    "std": "0.0",
                    "session_count": "1",
                    "window_count": "3",
                }
            )

    summaries = {row.group: row for row in summarize_template_rows(rows)}

    assert summaries["blink"].peak_relative_time == 0.0
    assert summaries["blink"].span == 1.0
    assert summaries["blink"].endpoint_delta == 0.0
    assert summaries["blink"].sign == "positive"
    assert summaries["background"].blink_distance > 0.0
