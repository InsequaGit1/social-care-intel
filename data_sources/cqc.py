"""
CQC Syndication API client.

Provides authoritative provider and location data from the Care Quality Commission.
This eliminates a major class of LLM hallucinations (made-up ratings, fake provider IDs).

API portal: https://api-portal.service.cqc.org.uk/
Base URL:   https://api.service.cqc.org.uk/public/v1
Auth:       Header `Ocp-Apim-Subscription-Key: {key}`

Strategy for "find provider by name" (the API has no name-search endpoint):
- Use Brave Search to find the CQC profile URL: site:cqc.org.uk "<name>"
- Extract provider_id (1-XXXXXX) or location_id (1-YYYYYY) from the URL
- Hit the CQC API directly to fetch authoritative data
"""

import re
from typing import Any, Dict, List, Optional

import requests

# Match CQC IDs in profile URLs, e.g.
#   /provider/1-101681838
#   /location/1-102643023
_CQC_ID_RX = re.compile(r"/(provider|location)/(1-\d+)", re.IGNORECASE)


class CQCClient:
    BASE_URL = "https://api.service.cqc.org.uk/public/v1"

    def __init__(self, api_key: str, timeout: int = 15):
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Ocp-Apim-Subscription-Key": api_key,
            "Accept": "application/json",
            "User-Agent": "social-care-intel/0.1",
        })

    # ------------------------------------------------------------------
    # Low-level endpoints
    # ------------------------------------------------------------------

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        return self._get(f"/providers/{provider_id}")

    def get_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        return self._get(f"/locations/{location_id}")

    def get_provider_locations(self, provider_id: str) -> List[Dict[str, Any]]:
        data = self._get(f"/providers/{provider_id}")
        if not data:
            return []
        return data.get("locationIds", []) or []

    # ------------------------------------------------------------------
    # High-level helpers used by the research pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def extract_id_from_url(url: str) -> Optional[Dict[str, str]]:
        """Pull (type, id) from a cqc.org.uk profile URL."""
        m = _CQC_ID_RX.search(url or "")
        if not m:
            return None
        return {"type": m.group(1).lower(), "id": m.group(2)}

    def fetch_from_url(self, cqc_url: str) -> Optional[Dict[str, Any]]:
        """
        Given a cqc.org.uk profile URL, return the authoritative data for it.
        Wraps location/provider lookups behind one call.
        """
        parsed = self.extract_id_from_url(cqc_url)
        if not parsed:
            return None
        if parsed["type"] == "location":
            data = self.get_location(parsed["id"])
            if data:
                data["_cqc_url"] = cqc_url
                data["_lookup_type"] = "location"
            return data
        elif parsed["type"] == "provider":
            data = self.get_provider(parsed["id"])
            if data:
                data["_cqc_url"] = cqc_url
                data["_lookup_type"] = "provider"
            return data
        return None

    def summarise_provider_profile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reduce a raw CQC payload (provider or location) to the fields the
        research pipeline cares about: overall rating, key dates, locations.
        """
        if not data:
            return {}

        # Both location and provider payloads include `currentRatings`
        current = data.get("currentRatings") or {}
        overall = (current.get("overall") or {}).get("rating", "Unknown")

        # Location-level inspection date
        last_inspection = data.get("lastInspection", {}) or {}
        inspection_date = last_inspection.get("date") or data.get("lastReport", {}).get("publicationDate")

        result = {
            "provider_id": data.get("providerId") or "",
            "location_id": data.get("locationId") or "",
            "name": data.get("providerName") or data.get("name") or "",
            "overall_rating": overall,
            "last_inspection_date": inspection_date or "Unknown",
            "registration_status": data.get("registrationStatus") or "",
            "registration_date": data.get("registrationDate") or "",
            "address": ", ".join(filter(None, [
                data.get("postalAddressLine1") or "",
                data.get("postalAddressLine2") or "",
                data.get("postalAddressTownCity") or "",
                data.get("postalCode") or "",
            ])),
            "region": data.get("region") or "",
            "local_authority": data.get("localAuthority") or "",
            "type": data.get("type") or "",
            "registered_manager": (data.get("relationships") or [{}])[0].get("relatedLocationId", "") if data.get("relationships") else "",
            "cqc_url": data.get("_cqc_url", ""),
            "lookup_type": data.get("_lookup_type", ""),
        }

        # Sub-ratings if present (Safe, Effective, Caring, Responsive, Well-led)
        sub_ratings = {}
        for key in ("safe", "effective", "caring", "responsive", "wellLed"):
            band = (current.get(key) or {}).get("rating")
            if band:
                sub_ratings[key] = band
        if sub_ratings:
            result["sub_ratings"] = sub_ratings

        return result

    # ------------------------------------------------------------------
    # HTTP helper
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
        try:
            r = self.session.get(
                f"{self.BASE_URL}{path}",
                params=params or {},
                timeout=self.timeout,
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            return None
