import json
from pathlib import Path

import pytest

from nisar_change import run_analysis
from nisar_change.errors import NoSuitableObservationsError, UnsupportedEventError
from nisar_change.events import METHODS, get_method

from helpers import granule, request, synthetic_pair


def test_unsupported_events_refuse_instead_of_reusing_deforestation():
    b, a = synthetic_pair()
    for name in ("wildfire", "flood", "earthquake_deformation"):
        m = get_method(name)
        assert not m.supported and m.requirements
        with pytest.raises(UnsupportedEventError, match="not implemented"):
            m.analyse(b, a, request(event_type=name))
    with pytest.raises(UnsupportedEventError, match="Unknown"):
        get_method("tornado")
    assert METHODS["deforestation"].supported


def _fakes():
    b, a = synthetic_pair()
    gs = [granule("BEFORE_G", "2026-10-25"), granule("AFTER_G", "2026-11-06")]
    return dict(
        search=lambda bbox, start, end: gs,
        cmr=lambda g: {"granule_concept_id": f"G-{g.id}", "collection_concept_id": "C1",
                       "start_time_matches_asf": True},
        retrieve=lambda g, bbox, pol, cache, info: (Path(g.id + ".tif"), {"job_id": "J1"}),
        reader=lambda path, units: b if str(path).startswith("BEFORE") else a,
    )


def test_end_to_end_with_synthetic_rasters(tmp_path):
    run = run_analysis(request(), tmp_path, write_geotiff=False, pixel_csv_max=500, **_fakes())
    for name in ("backscatter_csv", "classification_csv", "results_json", "metadata_json",
                 "pixels_csv"):
        assert run.paths[name].exists(), name
    results = json.loads(run.paths["results_json"].read_text())
    assert results["statistics"]["changed_pixels"] == 900
    meta = json.loads(run.paths["metadata_json"].read_text())
    assert meta["source_granules"]["before"]["id"] == "BEFORE_G"
    assert meta["polarisation"] == "HV" and meta["processing"]["thresholds"]["change_db"] == 3.0
    assert set(meta["apis_used"]) == {"ASF Search API", "NASA CMR", "NASA Harmony"}
    assert meta["limitations"] and meta["selection_explanation"]
    assert "password" not in json.dumps(meta).lower()
    assert len(run.paths["pixels_csv"].read_text().splitlines()) == 501


def test_no_observations_gives_error_and_no_files(tmp_path):
    fakes = _fakes()
    fakes["search"] = lambda bbox, start, end: []
    with pytest.raises(NoSuitableObservationsError):
        run_analysis(request(), tmp_path, write_geotiff=False, **fakes)
    assert not list(tmp_path.glob("*.csv"))


def test_unsupported_event_fails_before_any_network_call(tmp_path):
    def boom(*a, **k):
        raise AssertionError("network used")
    fakes = _fakes()
    fakes["search"] = boom
    with pytest.raises(UnsupportedEventError):
        run_analysis(request(event_type="flood"), tmp_path, write_geotiff=False, **fakes)
