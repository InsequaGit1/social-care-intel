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
        research pipeline cares about. Pulls the rich structured data CQC
        provides — sub-ratings, beds, registration date, service types,
        specialisms — so benchmarking can be grounded in real facts rather
        than LLM guesses from thin websites.
        """
        if not data:
            return {}

        current = data.get("currentRatings") or {}
        overall_obj = current.get("overall") or {}
        overall = overall_obj.get("rating") or ""
        rating_is_current = bool(overall)
        rating_report_date = overall_obj.get("reportDate") or ""

        # Fallback: many providers (esp. after a focused re-inspection) have an
        # empty currentRatings but a populated historicRatings — the CQC website
        # still shows the latest published rating from that history. Use the most
        # recent historic entry so we don't report "Unknown" when a rating exists.
        if not overall:
            historic = data.get("historicRatings") or []
            historic = sorted(historic, key=lambda h: h.get("reportDate", ""), reverse=True)
            for h in historic:
                h_overall = (h.get("overall") or {})
                if h_overall.get("rating"):
                    overall_obj = h_overall
                    overall = h_overall["rating"]
                    rating_report_date = h.get("reportDate") or ""
                    rating_is_current = False
                    break

        overall = overall or "Unknown"

        # Inspection / report date
        last_inspection = data.get("lastInspection", {}) or {}
        inspection_date = (
            last_inspection.get("date")
            or rating_report_date
            or (data.get("lastReport", {}) or {}).get("publicationDate")
        )

        website = data.get("website") or ""
        phone = data.get("phoneNumber") or ""

        # --- Sub-ratings (from currentRatings, or the historic entry we fell back to) ---
        sub_ratings = {}
        for kq in (overall_obj.get("keyQuestionRatings") or []):
            name = (kq.get("name") or "").strip()
            rating = kq.get("rating")
            if name and rating:
                sub_ratings[name] = rating

        # --- Service types, specialisms, regulated activities ---
        service_types = [
            (st.get("name") or "").strip()
            for st in (data.get("gacServiceTypes") or [])
            if st.get("name")
        ]
        specialisms = [
            (sp.get("name") or "").strip()
            for sp in (data.get("specialisms") or [])
            if sp.get("name")
        ]
        regulated_activities = [
            (ra.get("name") or "").strip()
            for ra in (data.get("regulatedActivities") or [])
            if ra.get("name")
        ]

        result = {
            "provider_id": data.get("providerId") or "",
            "location_id": data.get("locationId") or "",
            "name": data.get("providerName") or data.get("name") or "",
            "overall_rating": overall,
            "rating_is_current": rating_is_current,
            "rating_report_date": rating_report_date or "",
            "last_inspection_date": inspection_date or "Unknown",
            "registration_status": data.get("registrationStatus") or "",
            "registration_date": data.get("registrationDate") or "",
            "number_of_beds": data.get("numberOfBeds"),
            "website": website,
            "phone": phone,
            "address": ", ".join(filter(None, [
                str(data.get("postalAddressLine1") or ""),
                str(data.get("postalAddressLine2") or ""),
                str(data.get("postalAddressTownCity") or ""),
                str(data.get("postalCode") or ""),
            ])),
            "town": data.get("postalAddressTownCity") or "",
            "postcode": data.get("postalCode") or "",
            "region": data.get("region") or "",
            "local_authority": data.get("localAuthority") or "",
            "type": data.get("type") or "",
            "service_types": service_types,
            "specialisms": specialisms,
            "regulated_activities": regulated_activities,
            "cqc_url": data.get("_cqc_url", ""),
            "lookup_type": data.get("_lookup_type", ""),
        }
        if sub_ratings:
            result["sub_ratings"] = sub_ratings

        return result

    # ------------------------------------------------------------------
    # Area-based discovery — authoritative local market map
    # ------------------------------------------------------------------

    def list_locations(
        self,
        local_authority: str = "",
        region: str = "",
        care_home: Optional[bool] = None,
        per_page: int = 100,
        max_pages: int = 1,
    ) -> List[Dict[str, str]]:
        """
        List CQC locations filtered by local authority / region.
        Returns a list of {locationId, locationName}. This is the authoritative
        list of registered providers in an area — no LLM guessing.
        """
        params: Dict[str, Any] = {"perPage": per_page, "page": 1}
        if local_authority:
            params["localAuthority"] = local_authority
        if region:
            params["region"] = region
        if care_home is not None:
            params["careHome"] = "Y" if care_home else "N"

        locations: List[Dict[str, str]] = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self._get("/locations", params=params)
            if not data:
                break
            batch = data.get("locations") or []
            if not batch:
                break
            for loc in batch:
                lid = loc.get("locationId")
                if lid:
                    locations.append({
                        "locationId": lid,
                        "locationName": loc.get("locationName") or loc.get("name") or "",
                    })
            if len(batch) < per_page:
                break
        return locations

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
