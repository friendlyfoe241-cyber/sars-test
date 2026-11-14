"""Orchestration: request -> discovery -> selection -> retrieval -> alignment ->
analysis -> files.

Service roles
  ASF Search   discovery of granules (metadata)
  CMR          verification of the chosen granules + concept IDs/variable names (metadata)
  Harmony      spatial + variable subsetting and delivery of float GeoTIFF data
Credentials are only read inside retrieval.get_client().

The search / cmr / retrieve / reader callables are injectable so the
orchestration can be tested without network access.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

from . import discovery, events, outputs, raster, retrieval
from .analysis import ChangeResult
from .config import AnalysisRequest
from .errors import UnsupportedEventError
from .selection import SelectedPair, select_pair

API_ROLES = {
    "ASF Search API": "discovery of NISAR L2 GCOV granules and their metadata",
    "NASA CMR": "verification of selected granules; collection/granule concept IDs; variable names",
    "NASA Harmony": "bbox + polarisation-layer subsetting; delivers the float GeoTIFF data",
}


@dataclass
class RunResult:
    stats: dict
    paths: dict
    metadata: dict
    selection: SelectedPair | None
    result: ChangeResult


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "rasterio", "harmony"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def _require_supported(event_type: str):
    """Raise before any network or file work if the event has no implemented method."""
    method = events.get_method(event_type)
    if not method.supported:
        raise UnsupportedEventError(
            f"'{method.name}' analysis is not implemented, and the deforestation method must "
            f"not be reused for it. What it needs: {method.requirements}")
    return method


def _analyse(request: AnalysisRequest, before: raster.Raster, after: raster.Raster):
    method = _require_supported(request.event_type)
    after_aligned, align_info = raster.align_pair(
        before, after, request.resampling, request.min_overlap_fraction)
    result = method.analyse(before, after_aligned, request)
    return method, result, align_info


def _metadata(request, method, result, before, align_info, *, granules, selection_text,
              cmr, retrieval_info, mode) -> dict:
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "request": request.to_dict(),
        "location": {"bbox_west_south_east_north": list(request.bbox)},
        "observation_dates": {"requested_before": request.before_date.isoformat(),
                              "requested_after": request.after_date.isoformat()},
        "polarisation": request.polarisation,
        "source_granules": granules,
        "selection_explanation": selection_text,
        "cmr_verification": cmr,
        "retrieval": retrieval_info,
        "processing": {
            "event_method": method.name,
            "method_summary": method.summary,
            "input_units": request.input_units,
            "units_of_results": "dB (10*log10 of linear power)",
            "thresholds": {"change_db": request.thresholds.change_db,
                           "direction": request.thresholds.direction,
                           "db_valid_min": request.thresholds.db_valid_min,
                           "db_valid_max": request.thresholds.db_valid_max},
            "alignment": align_info,
            "analysis_grid": before.grid_info(),
            "speckle_filtering": "none",
        },
        "apis_used": API_ROLES if mode == "api" else {},
        "limitations": events.GENERAL_LIMITATIONS + method.limitations,
        "software": _versions(),
    }


def run_from_rasters(request: AnalysisRequest, before: raster.Raster, after: raster.Raster,
                     out_dir, *, pixel_csv_max=None, write_geotiff=True, granules=None,
                     selection_text=None, cmr=None, retrieval_info=None,
                     mode="rasters") -> RunResult:
    request.validate()
    method, result, align_info = _analyse(request, before, after)
    meta = _metadata(request, method, result, before, align_info, granules=granules,
                     selection_text=selection_text, cmr=cmr, retrieval_info=retrieval_info,
                     mode=mode)
    paths = outputs.write_all(result, before, meta, Path(out_dir), pixel_csv_max=pixel_csv_max,
                              write_geotiff=write_geotiff)
    return RunResult(result.stats, paths, meta, None, result)


def run_from_files(request: AnalysisRequest, before_tif, after_tif, out_dir, **kw) -> RunResult:
    """Compare two float GeoTIFFs you already have (e.g. earlier Harmony downloads)."""
    request.validate()
    _require_supported(request.event_type)
    before = raster.read_geotiff(before_tif, request.input_units)
    after = raster.read_geotiff(after_tif, request.input_units)
    return run_from_rasters(
        request, before, after, out_dir, mode="local_files",
        granules={"note": "local files supplied by the user; no catalogue metadata",
                  "before_file": str(before_tif), "after_file": str(after_tif)}, **kw)


def run_analysis(request: AnalysisRequest, out_dir, *, cache_dir=None, pixel_csv_max=None,
                 write_geotiff=True, search=discovery.search_asf, cmr=discovery.cmr_lookup,
                 retrieve=retrieval.retrieve_subset, reader=raster.read_geotiff) -> RunResult:
    """Full pipeline against the live NASA services."""
    request.validate()
    _require_supported(request.event_type)  # fail early for unsupported events (no network use)

    tol = timedelta(days=request.date_tolerance_days)
    start = datetime.combine(request.before_date, time.min, tzinfo=timezone.utc) - tol
    end = datetime.combine(request.after_date, time.max, tzinfo=timezone.utc) + tol
    granules = search(request.bbox, start, end)

    selection = select_pair(granules, request)
    cmr_info = {"before": cmr(selection.before), "after": cmr(selection.after)}
    warnings_ = [f"CMR start time for {role} granule differs from ASF's."
                 for role, info in cmr_info.items() if info.get("start_time_matches_asf") is False]

    cache = Path(cache_dir) if cache_dir else Path(out_dir) / ".cache"
    pol = request.polarisation
    tif_b, info_b = retrieve(selection.before, request.bbox, pol, cache, cmr_info["before"])
    tif_a, info_a = retrieve(selection.after, request.bbox, pol, cache, cmr_info["after"])
    before = reader(tif_b, request.input_units)
    after = reader(tif_a, request.input_units)

    text = selection.explanation + warnings_
    run = run_from_rasters(
        request, before, after, out_dir, pixel_csv_max=pixel_csv_max, write_geotiff=write_geotiff,
        granules={"before": selection.before.to_dict(), "after": selection.after.to_dict()},
        selection_text=text, cmr=cmr_info,
        retrieval_info={"before": info_b, "after": info_a}, mode="api")
    run.selection = selection
    return run
