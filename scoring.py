"""
Deterministic benchmarking scores.

Pure functions: given a company's authoritative CQC structured data (plus the
target's data for like-for-like comparison), return a 1-5 score for each
criterion with a transparent, verifiable justification.

Design goals:
  - Reproducible: same input → identical output, every run (no LLM, no network).
  - Verifiable: every score cites the exact CQC fact it used.
  - Consistent: the same rubric is applied to every company.

The LLM is NOT involved in scoring. It only writes the synthesis narrative
(executive summary + bid positioning) on top of these fixed scores.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

RATING_POINTS = {
    "outstanding": 5,
    "good": 4,
    "requires improvement": 2,
    "inadequate": 1,
}


def _clamp(n: int, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, n))


def _years_since(date_str: str) -> Optional[float]:
    """Years between an ISO-ish date string and today. None if unparseable."""
    if not date_str:
        return None
    s = str(date_str)[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            d = datetime.strptime(s, fmt).date()
            return (date.today() - d).days / 365.25
        except ValueError:
            continue
    # Year-only fallback
    try:
        y = int(s[:4])
        return date.today().year - y
    except (ValueError, TypeError):
        return None


def _rating_points(rating: str) -> Optional[int]:
    if not rating:
        return None
    return RATING_POINTS.get(str(rating).strip().lower())


def _norm_list(items: List[Any]) -> List[str]:
    return [str(x).strip().lower() for x in (items or []) if str(x).strip()]


# Official procurement domains — a contract only counts as EVIDENCED when its
# source_url sits on one of these (or another .gov.uk site).
_OFFICIAL_PROCUREMENT_DOMAINS = (
    "contractsfinder.service.gov.uk",
    "find-tender.service.gov.uk",
    "ted.europa.eu",
)


def validated_contracts(contracts: Optional[List]) -> List[Dict[str, Any]]:
    """
    Filter a contracts list down to entries with verifiable official sources.
    Deterministic evidence gate: LLM-claimed contracts without a real
    procurement URL are excluded from scoring (they may still be shown in the
    dashboard as unevidenced claims, but they never move a score).
    """
    out: List[Dict[str, Any]] = []
    for c in contracts or []:
        if not isinstance(c, dict):
            continue
        title = str(c.get("title") or "").strip().lower()
        # Negative findings recorded as entries ("No contracts found") don't count
        if title.startswith("no contract") or title.startswith("no award") or title.startswith("no record"):
            continue
        url = str(c.get("source_url") or "").strip()
        if not url.startswith("http"):
            continue
        host = url.split("/")[2].lower() if url.count("/") >= 2 else ""
        official = any(d in host for d in _OFFICIAL_PROCUREMENT_DOMAINS) or host.endswith(".gov.uk")
        if not official:
            continue
        path = url.split(host, 1)[-1].lower()
        # A search/listing page is not evidence of a specific contract
        if "/search" in path or "results" in path.rsplit("/", 1)[-1][:12]:
            continue
        # Reject placeholder/sequential notice IDs (hallucination fingerprints)
        if "1234567890" in url or "0987654321" in url:
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Individual criterion scorers — each returns (score, justification, basis)
# ---------------------------------------------------------------------------

def score_quality_compliance(cqc: Dict[str, Any]) -> Dict[str, Any]:
    """
    From CQC sub-ratings (Safe/Effective/Caring/Responsive/Well-led), or the
    overall rating if sub-ratings absent. Any Inadequate domain caps at 2;
    any Requires improvement caps at 3.
    """
    subs = cqc.get("sub_ratings") or {}
    pts = [_rating_points(v) for v in subs.values()]
    pts = [p for p in pts if p is not None]

    if pts:
        avg = sum(pts) / len(pts)
        score = round(avg)
        lows = [k for k, v in subs.items() if str(v).lower() == "inadequate"]
        ris = [k for k, v in subs.items() if str(v).lower() == "requires improvement"]
        if lows:
            score = min(score, 2)
        elif ris:
            score = min(score, 3)
        score = _clamp(score)
        sub_str = ", ".join(f"{k}: {v}" for k, v in subs.items())
        return {
            "score": score,
            "justification": f"Based on CQC sub-ratings ({sub_str}).",
            "source": cqc.get("cqc_url", ""),
            "basis": {"sub_ratings": subs},
        }

    overall = cqc.get("overall_rating") or cqc.get("rating")
    op = _rating_points(overall)
    if op is not None:
        return {
            "score": _clamp(op),
            "justification": f"Based on CQC overall rating ({overall}); sub-ratings not published.",
            "source": cqc.get("cqc_url", ""),
            "basis": {"overall_rating": overall},
        }

    return {
        "score": 2,
        "justification": "No CQC rating published (newly registered or not yet inspected).",
        "source": cqc.get("cqc_url", ""),
        "basis": {"overall_rating": "Unknown"},
    }


def score_delivery_strength(cqc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scale + longevity. Care homes use registered beds; domiciliary (beds=0)
    leans on longevity as a proxy for established delivery capacity.
    """
    beds = cqc.get("number_of_beds") or 0
    try:
        beds = int(beds)
    except (ValueError, TypeError):
        beds = 0
    years = _years_since(cqc.get("registration_date", "")) or 0

    score = 1
    reasons = []

    # Longevity component (max +2)
    if years >= 8:
        score += 2
        reasons.append(f"established {int(years)} years")
    elif years >= 3:
        score += 1
        reasons.append(f"{int(years)} years registered")

    # Scale component (max +2)
    if beds > 0:
        if beds >= 50:
            score += 2
            reasons.append(f"large capacity ({beds} beds)")
        elif beds >= 20:
            score += 1
            reasons.append(f"medium capacity ({beds} beds)")
        else:
            reasons.append(f"small capacity ({beds} beds)")
    else:
        # Domiciliary: no bed count; established agencies tend to have scale
        if years >= 8:
            score += 1
            reasons.append("established domiciliary agency")
        reasons.append("domiciliary (no bed capacity metric)")

    score = _clamp(score)
    return {
        "score": score,
        "justification": "Delivery scale/longevity: " + ("; ".join(reasons) if reasons else "limited public scale data") + ".",
        "source": cqc.get("cqc_url", ""),
        "basis": {"number_of_beds": beds, "registration_date": cqc.get("registration_date", "")},
    }


def score_service_location_fit(cqc: Dict[str, Any], target_cqc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same local authority + service-type overlap + specialism overlap with the
    target. Measures how directly this provider competes for the same work.
    """
    score = 1
    reasons = []

    comp_la = (cqc.get("local_authority") or "").strip().lower()
    targ_la = (target_cqc.get("local_authority") or "").strip().lower()
    if comp_la and targ_la and comp_la == targ_la:
        score += 1
        reasons.append(f"same local authority ({cqc.get('local_authority')})")
    elif comp_la:
        reasons.append(f"operates in {cqc.get('local_authority')}")

    comp_types = set(_norm_list(cqc.get("service_types")))
    targ_types = set(_norm_list(target_cqc.get("service_types")))
    if comp_types and targ_types and (comp_types & targ_types):
        score += 2
        reasons.append("same CQC service type(s)")
    elif comp_types and not targ_types:
        score += 1  # we couldn't compare, but it's a recognised social-care type
        reasons.append("recognised social-care provider")

    comp_spec = set(_norm_list(cqc.get("specialisms")))
    targ_spec = set(_norm_list(target_cqc.get("specialisms")))
    shared_spec = comp_spec & targ_spec
    if shared_spec:
        score += 1
        reasons.append(f"{len(shared_spec)} shared specialism(s)")

    score = _clamp(score)
    return {
        "score": score,
        "justification": "Service/location fit: " + ("; ".join(reasons) if reasons else "limited overlap") + ".",
        "source": cqc.get("cqc_url", ""),
        "basis": {
            "same_la": comp_la == targ_la and bool(comp_la),
            "service_overlap": sorted(comp_types & targ_types),
            "shared_specialisms": sorted(shared_spec),
        },
    }


def score_local_track_record(cqc: Dict[str, Any], target_cqc: Dict[str, Any],
                             contracts: Optional[List] = None) -> Dict[str, Any]:
    """
    Longevity in the target's local authority + any named contracts with the
    commissioner. Distinguishes "could bid" from "has actually delivered here".
    """
    score = 1
    reasons = []

    comp_la = (cqc.get("local_authority") or "").strip().lower()
    targ_la = (target_cqc.get("local_authority") or "").strip().lower()
    in_area = bool(comp_la and targ_la and comp_la == targ_la)
    years = _years_since(cqc.get("registration_date", "")) or 0

    if in_area:
        if years >= 8:
            score += 2
            reasons.append(f"established {int(years)} years in {cqc.get('local_authority')}")
        elif years >= 3:
            score += 1
            reasons.append(f"{int(years)} years in {cqc.get('local_authority')}")
        else:
            reasons.append(f"recently registered in {cqc.get('local_authority')}")
    else:
        reasons.append("not registered in the target's local authority")

    evidenced = validated_contracts(contracts)
    n_claimed = len(contracts or [])
    n_evidenced = len(evidenced)
    if n_evidenced:
        score += 2
        reasons.append(f"{n_evidenced} contract(s) evidenced on official procurement sources")
    elif n_claimed:
        reasons.append(f"{n_claimed} contract claim(s) lack official-source evidence — not counted")

    score = _clamp(score)
    return {
        "score": score,
        "justification": "Local track record: " + "; ".join(reasons) + ".",
        "source": (evidenced[0].get("source_url") if evidenced else cqc.get("cqc_url", "")),
        "basis": {"in_target_la": in_area, "evidenced_contracts": n_evidenced,
                  "claimed_contracts": n_claimed,
                  "registration_date": cqc.get("registration_date", "")},
    }


def score_strategic_differentiators(cqc: Dict[str, Any],
                                    website_evidence_quality: Optional[str] = None) -> Dict[str, Any]:
    """
    The least authoritative criterion — derived as a consistent proxy from CQC
    specialism breadth + service breadth, optionally strengthened by Deep-Scan
    website evidence. Flagged as indicative in the justification.
    """
    score = 1
    reasons = []

    specs = _norm_list(cqc.get("specialisms"))
    if len(specs) >= 6:
        score += 2
        reasons.append(f"broad specialism range ({len(specs)})")
    elif len(specs) >= 3:
        score += 1
        reasons.append(f"several specialisms ({len(specs)})")

    types = _norm_list(cqc.get("service_types"))
    if len(types) >= 2:
        score += 1
        reasons.append(f"multi-service provider ({len(types)} service types)")

    if website_evidence_quality:
        weq = website_evidence_quality.strip().lower()
        if weq == "strong":
            score += 1
            reasons.append("strong website evidence")
        elif weq in ("very weak", "weak"):
            reasons.append("weak website evidence")

    score = _clamp(score)
    return {
        "score": score,
        "justification": "Indicative (CQC specialism/service breadth"
                         + (" + website evidence" if website_evidence_quality else "")
                         + "): " + ("; ".join(reasons) if reasons else "limited differentiation signals") + ".",
        "source": cqc.get("cqc_url", ""),
        "basis": {"specialism_count": len(specs), "service_type_count": len(types),
                  "website_evidence_quality": website_evidence_quality or "n/a"},
    }


# Weights for the composite overall score (sum to 1.0)
OVERALL_WEIGHTS = {
    "cqc": 0.30,            # CQC rating quality
    "service_location_fit": 0.20,
    "delivery_strength": 0.20,
    "local_track_record": 0.20,
    "strategic_differentiators": 0.10,
}


def score_overall(scores: Dict[str, Dict], cqc_rating: str) -> Dict[str, Any]:
    """Weighted composite of the criteria + the CQC rating, rounded to 1-5."""
    cqc_pts = _rating_points(cqc_rating)
    if cqc_pts is None:
        cqc_pts = 2  # unrated treated as below-average, not zero
    total = OVERALL_WEIGHTS["cqc"] * cqc_pts
    for key in ("service_location_fit", "delivery_strength", "local_track_record",
                "strategic_differentiators"):
        total += OVERALL_WEIGHTS[key] * scores[key]["score"]
    raw = round(total, 2)
    score = _clamp(round(total))
    return {
        "score": score,
        "raw_score": raw,  # finer-grained for ranking / tie-breaking
        "justification": (
            f"Weighted composite = {raw}/5 — CQC {cqc_rating or 'Unrated'} ({cqc_pts}/5, 30%), "
            f"service/location fit {scores['service_location_fit']['score']}/5 (20%), "
            f"delivery {scores['delivery_strength']['score']}/5 (20%), "
            f"local track record {scores['local_track_record']['score']}/5 (20%), "
            f"differentiators {scores['strategic_differentiators']['score']}/5 (10%)."
        ),
        "source": "Composite of CQC-derived scores",
        "basis": {"weights": OVERALL_WEIGHTS, "raw_score": raw},
    }


def score_company(company: Dict[str, Any], target: Dict[str, Any],
                  contracts: Optional[List] = None,
                  website_evidence_quality: Optional[str] = None) -> Dict[str, Any]:
    """
    Produce the full deterministic score set for one company.

    `company` and `target` each provide a cqc_data dict (sub_ratings, beds,
    registration_date, service_types, specialisms, local_authority) plus
    cqc_rating / cqc_profile_url at the top level.
    """
    cqc = dict(company.get("cqc_data") or {})
    cqc.setdefault("cqc_url", company.get("cqc_profile_url", ""))
    cqc.setdefault("overall_rating", company.get("cqc_rating", "Unknown"))

    target_cqc = dict(target.get("cqc_data") or {})

    result: Dict[str, Any] = {}
    result["service_location_fit"] = score_service_location_fit(cqc, target_cqc)
    result["quality_compliance"] = score_quality_compliance(cqc)
    result["local_track_record"] = score_local_track_record(cqc, target_cqc, contracts)
    result["delivery_strength"] = score_delivery_strength(cqc)
    result["strategic_differentiators"] = score_strategic_differentiators(cqc, website_evidence_quality)
    result["overall_bid_threat"] = score_overall(result, company.get("cqc_rating", "Unknown"))

    # CQC rating as the authoritative word value
    result["cqc_rating"] = {
        "value": company.get("cqc_rating", "Unknown") or "Unknown",
        "url": company.get("cqc_profile_url", "") or "",
        "verified": bool(company.get("cqc_verified", False)),
        "is_current": bool(cqc.get("rating_is_current", True)),
        "report_date": cqc.get("rating_report_date", "") or "",
    }

    # Mark each scored criterion as deterministic + add a flat source field
    for k, v in result.items():
        if k == "cqc_rating":
            continue
        v["method"] = "deterministic (CQC-derived)"
        v.setdefault("analyst_inference", False)

    return result
