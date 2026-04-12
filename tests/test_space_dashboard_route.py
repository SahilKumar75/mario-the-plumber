from __future__ import annotations

import os
import subprocess
import sys

def test_space_root_route_serves_dashboard_html() -> None:
    script = """
from fastapi.testclient import TestClient
from app import app as space_app

response = TestClient(space_app).get("/")
assert response.status_code == 200
assert response.headers.get("content-type", "").startswith("text/html")
assert "<!doctype html" in response.text.lower()
print("ok")
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
