"""Shared daily API budget tracker for Open-Meteo.

Both historical and projection collection scripts share a single daily API
limit (~10,000 weighted calls/day). This module provides a shared budget
file so neither script over-estimates available quota.

Usage:
    from climate.budget import SharedBudget
    budget = SharedBudget()
    if budget.can_afford(887):
        # do the request
        budget.record(887, source="historical")
"""

import json
from datetime import datetime, timezone
from pathlib import Path

BUDGET_FILE = Path(__file__).parent.parent / "data" / "daily_budget.json"
DAILY_LIMIT = 9000  # Conservative: actual is ~10,000


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class SharedBudget:
    def __init__(self, budget_file: Path = BUDGET_FILE):
        self.path = budget_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = {
                "date": _today(),
                "calls_used": 0,
                "entries": [],
                "limit_hit": False,
            }
        # Auto-reset on new day
        if self._state.get("date") != _today():
            self._state = {
                "date": _today(),
                "calls_used": 0,
                "entries": [],
                "limit_hit": False,
            }
            self._save()

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.rename(self.path)

    @property
    def calls_used(self) -> int:
        return self._state.get("calls_used", 0)

    @property
    def remaining(self) -> int:
        return max(0, DAILY_LIMIT - self.calls_used)

    @property
    def limit_hit(self) -> bool:
        return self._state.get("limit_hit", False)

    def can_afford(self, cost: int) -> bool:
        """Check if there's enough budget for a request of given cost."""
        if self._state.get("limit_hit"):
            return False
        return self.calls_used + cost <= DAILY_LIMIT

    def record(self, cost: int, source: str = "unknown"):
        """Record API usage."""
        self._state["calls_used"] += cost
        self._state["entries"].append({
            "source": source,
            "cost": cost,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        })
        self._save()

    def mark_limit_hit(self, source: str = "unknown"):
        """Mark that the API returned a 429."""
        self._state["limit_hit"] = True
        self._state["limit_hit_at"] = datetime.now(timezone.utc).isoformat()
        self._state["entries"].append({
            "source": source,
            "event": "429_received",
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        })
        self._save()

    def clear_limit_hit(self):
        """Clear the limit_hit flag after a successful probe."""
        self._state["limit_hit"] = False
        self._state["entries"].append({
            "source": "probe",
            "event": "limit_cleared",
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        })
        self._save()

    def probe_api(self) -> bool:
        """Test if the API is accepting requests with a tiny probe.

        Makes a minimal request (1 location, 7 days, 1 variable) that costs
        only 1 weighted call. Returns True if the API responds with 200.
        """
        import httpx
        try:
            r = httpx.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": 51.51,
                    "longitude": -0.13,
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-07",
                    "daily": "temperature_2m_mean",
                },
                timeout=15,
            )
            if r.status_code == 200:
                self.clear_limit_hit()
                return True
            return False
        except Exception:
            return False

    def summary(self) -> str:
        return (f"Budget {self._state['date']}: {self.calls_used}/{DAILY_LIMIT} used, "
                f"{self.remaining} remaining, limit_hit={self.limit_hit}")
