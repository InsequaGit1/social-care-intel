"""
Offline logic tests — no network, no API keys required.

Run:  python3 test_logic.py

Validates the deterministic helper functions that underpin reliability:
  - CQC name disambiguation (right company, right town)
  - URL hallucination detection
  - CQC API response parsing (sub-ratings, beds, specialisms)
  - safe_join tolerance of mixed types

These are the parts that have repeatedly broken in production, so they
get explicit regression coverage.
"""

import sys

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"        got:  {got!r}")
        print(f"        want: {want!r}")
        FAILURES.append(name)


def check_true(name, cond):
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILURES.append(name)


def test_name_matching():
    print("\n== CQC name-match confidence ==")
    from research_agent import _name_match_confidence
    cases = [
        ("Ashley Care Ltd", "Ashley Care Limited", True),
        ("Ashley Care Ltd", "Ashley Community Care Services Limited", False),
        ("Bluebird Care", "Bluebird Care (Southend & Rochford)", True),
        ("Caremark", "Caremark Care Ltd", True),
        ("Mears Care", "Mears Group Ltd", False),
        ("Caremark", "Allied Healthcare", False),
    ]
    for query, candidate, want_accept in cases:
        score = _name_match_confidence(query, candidate)
        check(f"{query!r} ~ {candidate!r} -> accept={want_accept}", score >= 0.6, want_accept)


def test_url_filter():
    print("\n== Hallucinated URL detection ==")
    from research_agent import _is_url_suspicious
    fakes = [
        "https://www.contractsfinder.service.gov.uk/Notice/1234567890",
        "https://www.contractsfinder.service.gov.uk/Notice/1122334455",
        "https://www.contractsfinder.service.gov.uk/Notice/6677889900",
    ]
    reals = [
        "https://www.find-tender.service.gov.uk/Notice/005986-2026",
        "https://www.contractsfinder.service.gov.uk/Notice/2c4e7f2b-1a2b-4c3d-9e8f-abc123456789",
    ]
    for u in fakes:
        check(f"fake flagged: ...{u[-12:]}", _is_url_suspicious(u), True)
    for u in reals:
        check(f"real allowed: ...{u[-12:]}", _is_url_suspicious(u), False)


def test_cqc_parsing():
    print("\n== CQC response parsing (sub-ratings, beds, specialisms) ==")
    from data_sources.cqc import CQCClient
    client = CQCClient("dummy")
    # Realistic CQC location payload shape
    raw = {
        "locationId": "1-123456789",
        "providerId": "1-987654321",
        "name": "Sunnyfields Care Home",
        "registrationStatus": "Registered",
        "registrationDate": "2011-04-01",
        "numberOfBeds": 42,
        "website": "https://sunnyfields.example.co.uk",
        "postalAddressTownCity": "Southend-on-Sea",
        "postalCode": "SS1 1AA",
        "region": "East of England",
        "localAuthority": "Southend-on-Sea",
        "currentRatings": {
            "overall": {
                "rating": "Good",
                "reportDate": "2023-05-01",
                "keyQuestionRatings": [
                    {"name": "Safe", "rating": "Requires improvement"},
                    {"name": "Effective", "rating": "Good"},
                    {"name": "Caring", "rating": "Good"},
                    {"name": "Responsive", "rating": "Good"},
                    {"name": "Well-led", "rating": "Good"},
                ],
            }
        },
        "gacServiceTypes": [{"name": "Care home service with nursing"}],
        "specialisms": [{"name": "Dementia"}, {"name": "Caring for adults over 65 yrs"}],
        "regulatedActivities": [{"name": "Accommodation for persons who require nursing or personal care"}],
        "_cqc_url": "https://www.cqc.org.uk/location/1-123456789",
        "_lookup_type": "location",
    }
    s = client.summarise_provider_profile(raw)
    check("overall_rating", s["overall_rating"], "Good")
    check("number_of_beds", s["number_of_beds"], 42)
    check("registration_date", s["registration_date"], "2011-04-01")
    check("local_authority", s["local_authority"], "Southend-on-Sea")
    check("sub_ratings Safe", s.get("sub_ratings", {}).get("Safe"), "Requires improvement")
    check("sub_ratings count", len(s.get("sub_ratings", {})), 5)
    check("specialisms", s["specialisms"], ["Dementia", "Caring for adults over 65 yrs"])
    check("service_types", s["service_types"], ["Care home service with nursing"])
    check("website extracted", s["website"], "https://sunnyfields.example.co.uk")
    check("url id extraction", CQCClient.extract_id_from_url(raw["_cqc_url"]),
          {"type": "location", "id": "1-123456789"})


def test_safe_join():
    print("\n== safe_join tolerance ==")
    from analysis_agent import _safe_join
    check("mixed types", _safe_join([{"title": "A"}, "B", None, {"name": "C"}]), "A, B, C")
    check("empty", _safe_join([]), "")
    check("limit", _safe_join(["a", "b", "c"], limit=2), "a, b")


def test_enrichment_status():
    print("\n== enrichment status computation ==")
    from research_agent import _compute_enrichment_status
    comp = {
        "website": "https://x.co.uk",
        "cqc_profile_url": "https://www.cqc.org.uk/location/1-1",
        "companies_house_number": "09782291",
        "known_contracts_with_commissioner": [{"title": "X"}],
    }
    s = _compute_enrichment_status(comp, {})
    check("website_found", s["website_found"], True)
    check("cqc_found", s["cqc_found"], True)
    check("companies_house_found", s["companies_house_found"], True)
    check("contracts_found", s["contracts_found"], True)


if __name__ == "__main__":
    test_name_matching()
    test_url_filter()
    test_cqc_parsing()
    test_safe_join()
    test_enrichment_status()
    print()
    if FAILURES:
        print(f"❌ {len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    print("✅ All logic tests passed.")
