import unittest
from pathlib import Path

from app.models.schemas import (
    DetectionResult,
    JobAccepted,
)


class ApiContractTests(unittest.TestCase):
    def test_legacy_route_models_unchanged(self):
        self.assertSetEqual(set(JobAccepted.model_fields.keys()), {"job_id", "status", "message"})
        self.assertIn("events", DetectionResult.model_fields)
        self.assertIn("peak_confidence", DetectionResult.model_fields)

    def test_new_resource_routes_registered(self):
        root = Path(__file__).resolve().parents[1]
        main_py = (root / "app" / "main.py").read_text(encoding="utf-8")
        detect_py = (root / "app" / "api" / "detect.py").read_text(encoding="utf-8")
        health_py = (root / "app" / "api" / "health.py").read_text(encoding="utf-8")
        stats_py = (root / "app" / "api" / "stats.py").read_text(encoding="utf-8")
        incidents_py = (root / "app" / "api" / "incidents.py").read_text(encoding="utf-8")
        jobs_py = (root / "app" / "api" / "jobs.py").read_text(encoding="utf-8")

        # Legacy endpoints remain present.
        self.assertIn('"/detect-accident"', detect_py)
        self.assertIn('"/jobs/{job_id}"', detect_py)
        self.assertIn('"/health"', health_py)

        # New resources are wired in application startup.
        self.assertIn("app.include_router(stats.router, prefix=prefix)", main_py)
        self.assertIn("app.include_router(incidents.router, prefix=prefix)", main_py)
        self.assertIn("app.include_router(jobs.router, prefix=prefix)", main_py)

        # New endpoint paths exist with explicit route decorators.
        self.assertIn('"/overview"', stats_py)
        self.assertIn("@router.get(\n    \"\"", incidents_py)
        self.assertIn('"/{incident_id}"', incidents_py)
        self.assertIn("@router.get(\n    \"\"", jobs_py)
        self.assertIn('"/{job_id}/status"', jobs_py)
        self.assertIn('"/{job_id}/detections"', jobs_py)


if __name__ == "__main__":
    unittest.main()
