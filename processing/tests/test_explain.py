"""POST /api/explain-change on the analysis server (mocked Gemini upstream)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

import explain
import serve


VALID_BODY = {
    "studyArea": {"name": "Tonj floodplain wetlands", "country": "South Sudan",
                  "phenomenon": "Seasonal wetland inundation"},
    "acquisitionPair": {"band": "L-band", "productType": "L2 GCOV", "polarisation": "HV",
                        "temporalBaselineDays": 12,
                        "earlier": "2026-10-25T03:42:52Z", "later": "2026-11-06T03:42:53Z"},
    "dataset": {"status": "nisar_quicklook", "unit": "display DN",
                "limitations": ["quicklook values are display units"]},
    "statistics": {"unit": "display DN", "mean": -2.1, "median": -3.0, "validFraction": 0.967},
    "analysis": {
        "eventType": "deforestation",
        "acquisitionDates": {"before": "2026-10-25", "after": "2026-11-06"},
        "processing": {"eventMethod": "deforestation",
                       "methodParameters": {"change_db": 3.0, "direction": "decrease"},
                       "unitsOfResults": "dB (10*log10 of linear power)"},
        "statistics": {"mean_difference_db": -0.39, "median_difference_db": -0.37,
                       "valid_fraction": 0.99},
        "areaEstimates": {"valid_analysed_area_km2": 1098.1,
                          "candidate_changed_area_km2": 113.5,
                          "candidate_changed_percent_of_valid": 10.3},
    },
}


class GeminiStub(BaseHTTPRequestHandler):
    captured = None

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        GeminiStub.captured = {"url": self.path, "body": json.loads(self.rfile.read(length))}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps({
            "observed_signal": "Median change was -0.37 dB.",
            "possible_interpretation": "Could indicate seasonal change.",
            "supporting_evidence": "Mean change -0.39 dB below the 3 dB threshold.",
            "limitations": "Cause cannot be established.",
            "recommended_next_investigation": "Compare a longer time series.",
        })}]}}]}).encode())


@pytest.fixture()
def gemini(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), GeminiStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_BASE", f"http://127.0.0.1:{server.server_address[1]}/v1beta")
    yield
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture()
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def test_round_trip_with_analysis_facts(gemini, api_server):
    res = requests.post(f"{api_server}/api/explain-change", json=VALID_BODY, timeout=10)
    assert res.status_code == 200
    exp = res.json()["explanation"]
    assert exp["observedSignal"] == "Median change was -0.37 dB."
    assert exp["supportingEvidence"].startswith("Mean change")
    assert exp["recommendedNextInvestigation"]
    prompt = json.dumps(GeminiStub.captured["body"]["contents"])
    assert "candidateChangedAreaKm2" in prompt and "meanChangeDb" in prompt
    assert GeminiStub.captured["url"].endswith("/models/gemini-2.5-flash:generateContent?key=test-key")


def test_missing_key_503(api_server, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_BASE", raising=False)
    res = requests.post(f"{api_server}/api/explain-change", json=VALID_BODY, timeout=10)
    assert res.status_code == 503
    assert "GEMINI_API_KEY" in res.json()["error"]["message"]


def test_invalid_body_400(api_server):
    res = requests.post(f"{api_server}/api/explain-change", json={"statistics": {}}, timeout=10)
    assert res.status_code == 400


def test_get_rejected(api_server):
    res = requests.get(f"{api_server}/api/explain-change", timeout=10)
    assert res.status_code == 404  # only POST is routed


def test_prompt_never_contains_secrets(gemini):
    body = dict(VALID_BODY)
    body["statistics"]["unit"] = "display DN"
    out = explain.explain(body)
    assert out["explanation"]["limitations"] == "Cause cannot be established."
    prompt = json.dumps(GeminiStub.captured["body"])
    assert "test-key" not in json.dumps(GeminiStub.captured["body"]["contents"])
