"""Are the narrative's distance and duration claims true? (#177)

`#132` grades hallucination by **which places are named**. This grades **the
relations asserted between them**, as a third metric of the same measurement --
never a term added to the other two, because the denominators differ: those
two count place names, this counts claims.

The gap is real and `#132` cannot see it. Its own run produced

    "At 1:02 PM, you'll reach the Reichstag [...] A 64-minute walk brings you
     to the iconic Brandenburg Gate by 2:06 PM."

for two stops 280 m apart. Every place named was real, in the right city and
in the prompt, so `wrong_city` and `unshown` both read zero on a narrative
carrying a 16x error. `#173` fixed the cause -- the facts block gave arrival
clocks and nothing else, so the model subtracted one from the next and called
the remainder a walk -- and this module is what would have caught it.

Three references, and the decomposition is the point
----------------------------------------------------
A claim can be wrong in two unrelated ways, and reporting one number would
blame the model for both:

  stated   what the prompt's facts block said the leg was (`leg_minutes`,
           `leg_km`). Comparing the claim against this measures **the model**:
           did it repeat what it was given?
  actual   what the leg really costs, from `data/street_network/*.npz` --
           the real footpath network, not a straight line.
  harness  stated vs actual. This is not about the model at all.

The third column exists because the prompts these narratives answer are built
with `use_real_routing=False` (`hallucination.build_prompts`), so every leg the
narrator is told is a straight-line estimate. Measured over the catalogue, the
street network is a median 1.20x the straight line and 1.5-1.6x on legs under
500 m. A narrative that repeats its prompt perfectly is therefore *still*
understating the walk, and attributing that to the model would be false. The
shipped app plans with `use_real_routing=True` (`views/itinerary.py`, since
#160); the evaluation harness does not, and #132's committed cache is built on
it. Re-running the measurement under real routing would cost a full set of
generations, which `#177` explicitly rules out -- so the gap is reported here
rather than closed, and closing it is a decision for whoever reads this.

What counts as a claim
----------------------
  duration  a number of minutes attached to a travel verb -- "a 4-minute
            walk", "reached in seven minutes". Digits and words both.
  distance  "1.7 km", "300 m".
  vague     "a short walk", "a stroll", "steps away", "just around the
            corner", "nearby", "adjacent" -- no number at all.

`vague` is reported in its own column and never folded into the numeric rate,
following `#132`'s refusal to add `wrong_city` and `unshown` together. Grading
it needs a threshold for "short", and an arbitrary one would make the metric
arguable, so it is taken from a number the repo already committed to:
`travel_modes.HYBRID_WALK_THRESHOLD_KM = 1.2`, documented there as "roughly the
distance past which a typical traveler stops walking by choice (~16 min on
foot)". A vague proximity claim is counted false when the real street distance
between the two stops exceeds it.

What is not scored
------------------
A claim whose leg cannot be resolved -- the narrative describes a hop but the
paragraph structure does not say between which two stops -- is counted and
reported, never guessed at. `#175`'s lesson one measurement over: a cell that
cannot be graded must not be given a score.

Run:  python -m roamwise.evaluation.geographic_validation
Writes `geographic_validation.csv` (one row per claim) and
`geographic_validation_summary.csv` (per config). **Costs zero generations**:
it reads `hallucination_generations.json` and never calls a model, which
`test_the_geographic_validation_spends_no_quota` asserts.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from roamwise.agents.orchestrator import SYNTHESIS_SYSTEM
from roamwise.evaluation.hallucination import CACHE_PATH, build_prompts, load_cache
from roamwise.knowledge_graph.build_graph import GraphIndex
from roamwise.optimization.street_network import fetch_distance_duration_matrix
from roamwise.optimization.travel_modes import HYBRID_WALK_THRESHOLD_KM

HERE = Path(__file__).parent
CLAIMS_CSV = HERE / "geographic_validation.csv"
SUMMARY_CSV = HERE / "geographic_validation_summary.csv"

# A claim is right if it lands within the larger of an absolute floor and a
# relative band. The floor matters at the short end -- "a 4-minute walk" for a
# 5-minute leg is not an error a traveler could notice -- and the band keeps
# the same tolerance from being generous on a 40-minute one.
MINUTE_FLOOR, DISTANCE_FLOOR_KM, RELATIVE_BAND = 1.5, 0.15, 0.20

_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
          "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_NUM = r"(\d+(?:[.,]\d+)?|" + "|".join(_WORDS) + r")"
# Models write "4-minute" with a hyphen, an en dash or a non-breaking hyphen
# depending on the tokenizer, and the difference is invisible in a terminal.
_DASH = r"[\s‐-―−-]*"
# Ways of saying a leg happened. `reach` and `bring` are here because the
# narratives use them without naming a mode at all -- "the bunker is reached in
# seven minutes", "brings you to the gate" -- and a list of modes alone silently
# dropped those claims out of the denominator. Found by a test, not by reading
# output: the missing claims looked exactly like claims that were never made.
_TRAVEL = (r"walk|stroll|drive|ride|transit|journey|hop|away|from|"
           r"reach|bring|continue|onward|head to")
# "wait", "visit" and friends are durations too, and they are *correct* ones:
# #173 put them in the prompt on purpose. Counting them as travel claims is
# how a first pass at this measurement reported the day's total walking time
# as an invented leg.
_NOT_TRAVEL = r"wait|visit|explore|spend|stay|linger|enjoy|allow|pause|total|duration|open"

_DURATION = re.compile(_NUM + _DASH + r"(?:minute|min)s?", re.I)
_DISTANCE = re.compile(_NUM + _DASH + r"(?:kilometre|kilometer|km|metre|meter|m)\b", re.I)
_VAGUE = re.compile(
    r"\b(short walk|stroll|steps away|just around the corner|adjacent|nearby|"
    r"a stone's throw|within walking distance)", re.I)
_HEADING = re.compile(r"\*\*(.+?)\*\*")


def _value(token: str) -> float:
    token = token.replace(",", ".")
    return float(token) if token[0].isdigit() else float(_WORDS[token.lower()])


def _sentence_around(text: str, start: int, end: int) -> str:
    """The sentence a match sits in.

    A fixed character window was tried first and is wrong: 60 characters after
    "Only a four-minute walk away, this monument is next." reach into the next
    paragraph's "This route totals 7 hours...", whose "total" is on the
    not-travel list, and a correct leg claim was silently dropped. Sentence
    bounds are what the disqualifying words actually scope over.
    """
    left = max((text.rfind(c, 0, start) for c in ".!?\n"), default=-1)
    right = min((r for r in (text.find(c, end) for c in ".!?\n") if r != -1),
                default=len(text))
    return text[left + 1:right]


def extract_claims(text: str) -> list[dict]:
    """Every distance/duration/proximity claim in one narrative.

    Context, not just the match, decides whether a duration is about travel:
    the sentence it sits in has to name a way of moving and must not name a
    visit or a wait. That is the whole difference between "a 4-minute walk"
    and "with a visit duration of 60 minutes", and both appear in these
    narratives because #173 put both in the prompt.
    """
    claims = []
    for kind, pattern in (("duration", _DURATION), ("distance", _DISTANCE)):
        for match in pattern.finditer(text or ""):
            window = _sentence_around(text, match.start(), match.end())
            if not re.search(_TRAVEL, window, re.I) or re.search(_NOT_TRAVEL, window, re.I):
                continue
            value = _value(match.group(1))
            if kind == "distance" and re.search(r"\b(metre|meter|m)\b$", match.group(0), re.I):
                value /= 1000.0            # normalise everything to km
            claims.append({"kind": kind, "value": value, "at": match.start(),
                           "phrase": " ".join(match.group(0).split())})
    for match in _VAGUE.finditer(text or ""):
        claims.append({"kind": "vague", "value": None, "at": match.start(),
                       "phrase": " ".join(match.group(0).split())})
    return claims


def resolve_leg(claim: dict, text: str, stop_names: list[str]) -> int | None:
    """Which stop the claim is about, as an index into the day's stop list.

    Returns (index, how), where `how` names the rule that resolved it: a bold
    heading is a much stronger signal than a nearby mention, and pooling the
    two silently would hide how much of the strict rate rests on the weaker
    one.

    The leg is then (stop - 1 -> stop). Resolution prefers the nearest bold
    heading above the claim, because these narratives are written as one
    paragraph per stop under its own name; it falls back to the nearest stop
    named earlier in the text. Returns None when neither resolves to a stop
    with a leg before it, and the caller must leave that claim ungraded rather
    than attach it to a guess.
    """
    headings = [(m.start(), m.group(1).strip(" *–—-")) for m in _HEADING.finditer(text)]
    above = [name for pos, name in headings if pos <= claim["at"]]
    if above:
        for index, name in enumerate(stop_names):
            if name and name in above[-1]:
                return (index, "heading") if index > 0 else (None, "first_stop")
    seen = {index: text.find(name) for index, name in enumerate(stop_names)
            if name and name in text}
    prior = [index for index, at in seen.items() if 0 <= at <= claim["at"]]
    if not prior:
        return None, "unresolved"
    index = max(prior, key=lambda i: seen[i])
    return (index, "nearest_mention") if index > 0 else (None, "first_stop")


def street_legs(poi_ids: list[str], idx: GraphIndex) -> list[tuple]:
    """Real (km, minutes) for each consecutive leg, from the committed street
    network. `None` for a leg the network cannot price -- a POI that failed to
    snap -- so the caller drops it rather than falling back to a straight line,
    which is the very thing this module exists to check against."""
    points = [{"lat": idx.g.nodes[p]["lat"], "lon": idx.g.nodes[p]["lon"]} for p in poi_ids]
    got = fetch_distance_duration_matrix(points, "foot")
    if got is None:
        return [None] * len(poi_ids)
    km, minutes = got
    return [None] + [(km[i - 1][i], minutes[i - 1][i]) for i in range(1, len(poi_ids))]


def _within(claimed: float, reference: float, floor: float) -> bool:
    return abs(claimed - reference) <= max(floor, RELATIVE_BAND * reference)


def matches_any_leg(claim: dict, schedule: list) -> bool | None:
    """Whether the claim equals *some* leg the prompt stated, whichever one.

    This is the measurement's check on itself. `vs_prompt` grades a claim
    against the one leg `resolve_leg` attributed it to, so a wrong attribution
    reads as an unfaithful model. This column cannot be wrong that way: a
    claim matching no leg at all is one no attribution could have saved. The
    distance between the two columns is how much of the strict number is this
    file rather than the narrator, and it belongs in the report -- #173
    measured 97.8% of travel-duration claims matching some stated leg, and a
    strict rate far below that while this column stays high would be a fact
    about the resolver, not about the model.
    """
    if claim["kind"] == "vague" or not schedule:
        return None
    key = "leg_minutes" if claim["kind"] == "duration" else "leg_km"
    floor = MINUTE_FLOOR if claim["kind"] == "duration" else DISTANCE_FLOOR_KM
    values = [slot[key] for slot in schedule
              if slot.get(key) is not None and slot[key] > 0]
    return any(_within(claim["value"], v, floor) for v in values) if values else None


def grade(claim: dict, stated: dict, actual: tuple) -> dict:
    """One claim against both references. Either can be absent, and absence is
    reported rather than scored."""
    out = {"vs_prompt": None, "vs_street": None}
    if claim["kind"] == "vague":
        # Only the ground truth can settle "short": the prompt's own estimate
        # is the thing under suspicion here, not the yardstick.
        if actual:
            out["vs_street"] = actual[0] <= HYBRID_WALK_THRESHOLD_KM
        return out
    floor = MINUTE_FLOOR if claim["kind"] == "duration" else DISTANCE_FLOOR_KM
    key = "leg_minutes" if claim["kind"] == "duration" else "leg_km"
    if stated and stated.get(key) is not None:
        out["vs_prompt"] = _within(claim["value"], stated[key], floor)
    if actual:
        out["vs_street"] = _within(claim["value"],
                                   actual[1] if claim["kind"] == "duration" else actual[0],
                                   floor)
    return out


def run(top_k: int = 8) -> tuple:
    """Returns (per-claim frame, per-config summary). Reads the committed
    generation cache; never calls a model."""
    idx = GraphIndex()
    cache = load_cache()
    prompts = build_prompts(top_k=top_k, with_itinerary=True)

    rows, missing = [], 0
    for _, row in prompts.iterrows():
        model = next((v["model"] for v in cache.values()), None)
        key = hashlib.sha1(
            f"{model}\x00{SYNTHESIS_SYSTEM}\x00{row.prompt}".encode()).hexdigest()
        if key not in cache:
            missing += 1
            continue
        text = cache[key]["text"] or ""
        names = [idx.g.nodes[p].get("name") for p in row.stops]
        legs = street_legs(row.stops, idx)
        for claim in extract_claims(text):
            index, how = resolve_leg(claim, text, names)
            stated = row.schedule[index] if index is not None and index < len(row.schedule) else None
            actual = legs[index] if index is not None and index < len(legs) else None
            rows.append({"query_id": row.query_id, "config": row.config,
                         "destination_id": row.destination_id, "tier": row.tier,
                         "kind": claim["kind"], "phrase": claim["phrase"],
                         "claimed": claim["value"],
                         "target": names[index] if index is not None else None,
                         "stated": (stated or {}).get(
                             "leg_minutes" if claim["kind"] == "duration" else "leg_km"),
                         "street_km": actual[0] if actual else None,
                         "street_min": actual[1] if actual else None,
                         "resolved_by": how,
                         "vs_any_leg": matches_any_leg(claim, row.schedule),
                         **grade(claim, stated, actual)})
    if missing:
        print(f"note: {missing} of {len(prompts)} prompts had no cached generation")

    claims = pd.DataFrame(rows)
    return claims, summarize(claims)


def summarize(claims: pd.DataFrame) -> pd.DataFrame:
    """Per config: how many claims, how many the model got right against its
    own prompt, and how many are true of the real street network.

    `n` is claims, not place names -- the whole reason this is a third metric
    and not a term in the other two.
    """
    out = []
    for config, group in claims.groupby("config"):
        numeric = group[group.kind != "vague"]
        vague = group[group.kind == "vague"]
        graded_prompt = numeric[numeric.vs_prompt.notna()]
        graded_street = numeric[numeric.vs_street.notna()]
        graded_vague = vague[vague.vs_street.notna()]
        out.append({
            "config": config,
            "claims": len(group),
            "numeric_claims": len(numeric),
            "vague_claims": len(vague),
            "unresolved": int(group.target.isna().sum()),
            "faithful_to_prompt": round(graded_prompt.vs_prompt.mean(), 4) if len(graded_prompt) else None,
            "n_vs_prompt": len(graded_prompt),
            "true_of_street_network": round(graded_street.vs_street.mean(), 4) if len(graded_street) else None,
            "n_vs_street": len(graded_street),
            "matches_some_stated_leg": round(numeric[numeric.vs_any_leg.notna()].vs_any_leg.mean(), 4)
            if numeric.vs_any_leg.notna().any() else None,
            "faithful_when_resolved_by_heading": round(
                graded_prompt[graded_prompt.resolved_by == "heading"].vs_prompt.mean(), 4)
            if (graded_prompt.resolved_by == "heading").any() else None,
            "vague_true_of_street_network": round(graded_vague.vs_street.mean(), 4) if len(graded_vague) else None,
            "n_vague_graded": len(graded_vague),
        })
    return pd.DataFrame(out)


def harness_gap(claims: pd.DataFrame) -> dict:
    """How far the prompt's own straight-line legs sit from the street network.

    Reported separately from everything else because it is not a property of
    the narrator: it is what `use_real_routing=False` costs, and it bounds how
    right a perfectly faithful narrative can be.
    """
    legs = claims[(claims.kind == "duration") & claims.stated.notna()
                  & claims.street_min.notna()].drop_duplicates(["query_id", "config", "target"])
    if legs.empty:
        return {}
    ratio = legs.street_min / legs.stated.replace(0, np.nan)
    return {"legs": len(legs),
            "median_street_over_stated": round(float(ratio.median()), 3),
            "p90_street_over_stated": round(float(ratio.quantile(0.9)), 3),
            "stated_within_band_of_street": round(float(
                (abs(legs.stated - legs.street_min)
                 <= np.maximum(MINUTE_FLOOR, RELATIVE_BAND * legs.street_min)).mean()), 4)}


if __name__ == "__main__":
    claims, summary = run()
    claims.to_csv(CLAIMS_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nRead {CACHE_PATH.name}; 0 generations.\n")
    print(summary.to_string(index=False))
    print("\nHarness gap (the prompt's straight-line legs vs the street network):")
    print(json.dumps(harness_gap(claims), indent=2))
