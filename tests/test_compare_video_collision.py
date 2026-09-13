import pytest
from test_compare_video import module


def test_existing_output_cannot_leave_stale_success_manifest(tmp_path):
    out = tmp_path/'existing'
    out.mkdir()
    marker = out/'manifest.json'
    marker.write_text('old result')
    with pytest.raises(FileExistsError):
        module.compare(tmp_path/'missing.avi', tmp_path/'also-missing.avi', 0, out)
    assert marker.read_text() == 'old result'
