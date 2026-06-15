from hp_acoustic_wave.audio_devices import resolve_audio_device_selection


def test_audio_device_selection_tolerates_missing_metadata_index():
    selection = resolve_audio_device_selection(
        requested_input_device=0,
        requested_output_device=1,
        default_device=[0, 1],
        devices=(),
        hostapis=(),
    )

    assert selection["resolved"]["input"] is None
    assert selection["resolved"]["output"] is None
