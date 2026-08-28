"""What a stop is worth to *this* traveler (issue #72).

The router's model today does not have a notion of a stop being worth more
than another: `optimize_day_route` keeps whatever the geographic ordering
reaches in time, so the decision is made by proximity and the clock alone.
TOPTW needs a per-POI score, because its whole objective is to maximise the
score collected -- and the score is where personalization belongs.

The issue's formulation:

    score(poi) = preference match x quality x crowding discount x diversity

with the requirement that the preference vector is used **directly**, not
collapsed to an archetype label first. That distinction is the point. Today
six sliders become one of seven labels, the label picks a retrieval query,
and everything the sliders said about *degree* is discarded: a traveler at
culture 0.95 and a traveler at culture 0.55 both come back "Culture
Enthusiast" and get the same itinerary. Here the sliders weight the score
themselves, so those two travelers get different ones.

Three of the four factors were implemented here from the start. The fourth,
the crowding discount, was a stub returning 1.0 until issue #71 put a per-POI
hourly series in the catalogue. Both halves of it are live now: this module
owns *which* stop -- a static node weight, which is all a selector can be --
and issue #33 gave *when* to the solver, which prices the hour by offering a
POI a second node pinned to its quietest three hours (`toptw._crowd_slots`).
The diversity factor lives in the solver rather than here too, for the reason
given in `score_pois`.

WHAT MEASURING IT SHOWED (evaluation/toptw_scoring_ablation.py). The score
has two possible jobs, and it is good at one of them:

  pool        weights    stops/day   km/stop    pref    qual  cat/day
  retrieval   uniform         8.00     0.362   0.614   0.722     2.83
  retrieval   scored          7.58     0.741   0.639   0.784     2.67
  score       uniform         8.33     0.360   0.606   0.791     3.08
  score       scored          7.83     0.624   0.628   0.792     2.92

As a **candidate selector** it earns its place: ranking the catalogue by
score directly, with no archetype label anywhere, matches what
`archetype_query` retrieves and slightly beats it -- more stops a day at the
same distance per stop, better-known places, more categories in a day.

As a **solver weight** on top of an already-selected pool it does not. It
costs 5-6% of the stops and doubles km/stop, and buys 0.614 -> 0.639 of
preference match -- which is its own objective, not an independent judge. A
term that can only move the very quantity it maximises by 4%, while paying
that much for it, is double-counting: retrieval (or the score, as selector)
has already spent the preference signal, and what is left for the weights to
act on is mostly fame, which trades directly against geometry.

So the recommendation this module carries into the TOPTW router: score the
candidates, then let the solver optimise geometry and time uniformly over
what the score chose.
"""
import math

import numpy as np
import pandas as pd

from roamwise.knowledge_graph.build_graph import (
    CATEGORY_AFFINITY, DATA_DIR, PROMINENCE_FLOOR)

# The traveler's own six sliders, in the order the survey stores them.
PREFERENCE_DIMS = ["budget", "culture", "nature", "nightlife", "relax", "adventure"]


def _fit_preference_matrix() -> dict[str, dict[str, float]]:
    """Preference dimension -> category weight, fitted rather than invented.

    The repo already holds both halves of this mapping, just never joined:
    `user_survey.csv` says what preference vector each archetype has, and
    `CATEGORY_AFFINITY` says what categories each archetype wants. Fitting
    preferences -> categories through the archetypes gives a matrix that
    cannot silently disagree with either table, and that keeps agreeing with
    them if someone retunes one (see MEMORY: 7 archetypes and 10 categories
    are load-bearing).

    Non-negative least squares, not plain least squares. Plain lstsq reports
    R^2 = 0.92 but reaches it with 28 of 60 weights negative -- "wanting
    nature makes you want museums less" -- and clipping those to zero
    collapses the fit to R^2 = -82, with all seven archetypes predicting the
    same top three categories. Constraining non-negativity up front costs
    fitted variance (R^2 = 0.60) and buys a mapping that means something: it
    recovers the correct top-3 categories in 16 of 21 archetype slots, and
    the right first choice for all but one.

    One honest gap it exposes: `adventure` fits to all zeros. The catalogue
    has no adventure-ish category of its own (`beach` holds nothing in the
    two-city dataset), so the slider's signal is entirely absorbed by
    `nature`, which it correlates with across the survey. The adventure
    slider therefore does not move this score on its own today.
    """
    from scipy.optimize import nnls

    survey = pd.read_csv(DATA_DIR / "user_survey.csv")
    archetypes = sorted(CATEGORY_AFFINITY)
    categories = sorted({c for a in CATEGORY_AFFINITY.values() for c in a})
    prefs = np.array([survey[survey.archetype == a][PREFERENCE_DIMS].mean().values
                      for a in archetypes])
    wants = np.array([[CATEGORY_AFFINITY[a].get(c, 0.0) for c in categories]
                      for a in archetypes])
    fitted = np.column_stack([nnls(prefs, wants[:, j])[0]
                              for j in range(len(categories))])
    return {dim: {cat: float(fitted[i, j]) for j, cat in enumerate(categories)}
            for i, dim in enumerate(PREFERENCE_DIMS)}


PREFERENCE_CATEGORY_WEIGHTS = _fit_preference_matrix()
# The largest match any slider setting can produce, used to normalise into
# [0, 1] so the factor is comparable with the others.
_MAX_MATCH = max(sum(PREFERENCE_CATEGORY_WEIGHTS[d][c] for d in PREFERENCE_DIMS)
                 for c in next(iter(PREFERENCE_CATEGORY_WEIGHTS.values())))
# What a POI in a category the traveler expressed no interest in is still
# worth. Not zero: a score of zero makes a stop free to drop whatever else is
# true of it, and an itinerary that visits *only* the favourite category is
# the same lexicographic starvation PROMINENCE_FLOOR exists to prevent
# upstream (see score_by_affinity_and_prominence).
PREFERENCE_FLOOR = 0.15


def preference_match(preferences: dict, category: str) -> float:
    """How much this traveler's sliders ask for this category, in [0, 1].

    Reads the vector directly: no archetype label is computed, looked up or
    consulted anywhere in this function."""
    weights = {d: PREFERENCE_CATEGORY_WEIGHTS[d].get(category, 0.0)
               for d in PREFERENCE_DIMS}
    raw = sum(float(preferences.get(d, 0.0)) * w for d, w in weights.items())
    return PREFERENCE_FLOOR + (1.0 - PREFERENCE_FLOOR) * min(raw / _MAX_MATCH, 1.0)


def quality(pois: list[dict]) -> list[float]:
    """How well-known each POI is, in [PROMINENCE_FLOOR, 1].

    Deliberately the same shape as `score_by_affinity_and_prominence`'s
    prominence half -- min-max normalised `popularity_score` lifted onto
    PROMINENCE_FLOOR -- because that floor was swept against iconic coverage
    and recall (#63) and there is no reason for the router to disagree with
    retrieval about how much fame is worth."""
    scores = [p.get("popularity_score", 0.0) or 0.0 for p in pois]
    lo, hi = (min(scores), max(scores)) if scores else (0.0, 0.0)
    span = hi - lo
    return [PROMINENCE_FLOOR + (1.0 - PROMINENCE_FLOOR) *
            ((s - lo) / span if span else 1.0) for s in scores]


_CROWDING_CACHE: dict | None = None
# How far a crowding discount may move a stop's worth. At 0.35 the emptiest
# POI in the catalogue keeps 96% of its score and the busiest 72%, which is a
# real ordering effect without letting crowding overrule what the traveler
# actually asked for -- preference match and quality both span a wider range
# than that, and should.
CROWDING_STRENGTH = 0.35


def _crowding_tables() -> dict:
    """`data/crowding.csv` as the two shapes the callers want, loaded once.

    `typical` is the mean over the hours a POI shows any activity, not over
    the whole day: a venue open four hours would otherwise look empty because
    twenty of its twenty-four readings are zeros about being shut.

    `by_category` is the fallback, and having one is the point. Scoring an
    unmeasured POI at 1.0 would hand every place the enrichment missed a
    silent bonus over the places it measured -- the busiest POIs would be
    penalised and the unknown ones would not, which rewards absence of data.
    A category's own mean is the least-assuming stand-in the catalogue can
    offer."""
    global _CROWDING_CACHE
    if _CROWDING_CACHE is not None:
        return _CROWDING_CACHE
    path = DATA_DIR / "crowding.csv"
    if not path.exists():
        _CROWDING_CACHE = {"hourly": {}, "typical": {}, "by_category": {}, "overall": None}
        return _CROWDING_CACHE
    df = pd.read_csv(path)
    hourly = {}
    for (poi_id, day, hour), busy in df.set_index(["poi_id", "day", "hour"])["busy"].items():
        hourly.setdefault(poi_id, {})[(day, int(hour))] = int(busy)
    active = df[df["busy"] > 0]
    typical = active.groupby("poi_id")["busy"].mean().to_dict()
    pois = pd.read_csv(DATA_DIR / "poi.csv", usecols=["poi_id", "category"])
    joined = pois[pois["poi_id"].isin(typical)].copy()
    joined["typical"] = joined["poi_id"].map(typical)
    _CROWDING_CACHE = {
        "hourly": hourly,
        "typical": typical,
        "by_category": joined.groupby("category")["typical"].mean().to_dict(),
        "overall": float(joined["typical"].mean()) if len(joined) else None,
    }
    return _CROWDING_CACHE


def busyness(poi: dict, day: str = None, hour: float = None) -> float | None:
    """How busy this POI is, 0-100, or None when nothing is known about it.

    With no day and hour, the POI's typical level across the hours it is open.
    With both, that hour's own reading -- which is the shape issue #72 asked
    for, and which is five times the signal the typical level carries: across
    the catalogue a POI's day swings a median 72 points from trough to peak,
    while POIs differ from each other by a standard deviation of 13."""
    tables = _crowding_tables()
    poi_id = poi.get("poi_id")
    if day is not None and hour is not None:
        series = tables["hourly"].get(poi_id)
        if series:
            reading = series.get((day, int(hour)))
            if reading is not None:
                return float(reading)
        return None
    if poi_id in tables["typical"]:
        return float(tables["typical"][poi_id])
    return None


def busyness_over(poi: dict, day: str, start_hour: float, end_hour: float) -> float | None:
    """Mean busyness over the open hours this POI has in `[start_hour, end_hour)`.

    The router needs a level it can attach to a *slot*, not to a clock
    reading. Which slot a stop falls in is a choice the model can make -- it
    is which node it activates -- while the arrival minute inside that slot is
    settled afterwards by everything else in the day. The mean over the slot
    is the level that does not pretend to know that minute.

    Hours reading 0 are skipped for the reason `_crowding_tables` gives: a
    zero is a statement about being shut, not about being empty. Returns None
    when the slot holds no reading at all, which is the same answer as "this
    POI was never measured" and is treated the same way by callers.
    """
    series = _crowding_tables()["hourly"].get(poi.get("poi_id"))
    if not series:
        return None
    hours = range(int(math.floor(start_hour)), int(math.ceil(end_hour)))
    readings = [series[(day, h)] for h in hours
                if series.get((day, h)) is not None and series[(day, h)] > 0]
    return sum(readings) / len(readings) if readings else None


def expected_busyness(poi: dict, day: str = None, start_hour: float = None,
                      end_hour: float = None) -> float:
    """Busyness for a POI the caller must price whether or not it was measured.

    `busyness` and `busyness_over` answer None when nothing is known, which is
    the right answer to a question about data. A router cannot use it: a stop
    it cannot price is a stop it prices at zero, and zero is the cheapest
    number there is -- so every POI the enrichment missed would become the
    cheapest hour of the day to visit. The ladder is the same one
    `crowding_discount` climbs, and for the same reason: the slot's own
    reading, then the POI's typical level, then its category's mean.
    """
    level = None
    if start_hour is not None and end_hour is not None:
        level = busyness_over(poi, day, start_hour, end_hour)
    if level is None:
        level = busyness(poi)
    if level is None:
        tables = _crowding_tables()
        level = tables["by_category"].get(poi.get("category"), tables["overall"])
    return 0.0 if level is None else float(level)


def crowding_discount(poi: dict = None, day: str = None, hour: float = None,
                      forecast: dict = None) -> float:
    """What a stop is worth once you account for the crowd it will be in.

    This was a constant 1.0 until issue #71 put a per-POI hourly series in the
    catalogue (`data/crowding.csv`, 269 POIs). The reason it had to be a stub
    is worth keeping: `ForecasterAgent` answers per city and per month, and a
    single city-month scalar multiplied onto every candidate rescales the whole
    pool identically, so it cannot change which stops an optimiser picks. It is
    still not read here, for exactly that reason -- `forecast` is accepted so
    callers need not change, and ignored.

    What the series buys is variation *between* POIs, which does survive
    normalisation: the emptiest POI in the catalogue averages 12% busy and the
    fullest 80%. A POI with no reading falls back to its category's mean rather
    than to no discount at all; see `_crowding_tables`.

    Passing `day` and `hour` gives the hour-resolved answer. This function is
    still asked without one, and deliberately: a score is a static node weight,
    so the hour cannot be an input to it. The router reaches the hour-resolved
    series by a different route -- `toptw._crowd_slots` turns "when" into a
    choice between nodes, which is a thing a static weight *can* price (#33).
    """
    if poi is None:
        return 1.0
    level = busyness(poi, day, hour)
    if level is None:
        tables = _crowding_tables()
        level = tables["by_category"].get(poi.get("category"), tables["overall"])
    if level is None:
        return 1.0
    return 1.0 - CROWDING_STRENGTH * (level / 100.0)


def score_pois(pois: list[dict], preferences: dict, forecast: dict = None) -> list[float]:
    """Per-POI worth, normalised to mean 1.0 over the pool.

    Normalised because the solver spends these against a drop penalty
    denominated in metres: holding the mean at 1.0 keeps "what a stop is
    worth" on the same scale whatever the traveler's sliders say, so the
    penalty sweep means the same thing across travelers and only the
    *relative* worth of stops changes.

    The diversity factor from the issue's formula is not here, and cannot be:
    "how many of this category are already in this day" is a property of a
    partial solution, not of a POI, so it has no value at the time a static
    node weight is set. It is expressed in the solver instead, as a per-day
    category cap -- which is the constraint-model form of the same rule, and
    one of the things the issue notes CP-SAT-style modelling buys.
    """
    if not pois:
        return []
    fame = quality(pois)
    raw = [preference_match(preferences, p.get("category")) * q * crowding_discount(p)
           for p, q in zip(pois, fame)]
    mean = sum(raw) / len(raw)
    return [r / mean for r in raw] if mean else [1.0] * len(raw)


# How many candidates to keep per day of the trip. The solver is comfortable
# to roughly 120 POIs and the working set has to hold meals too, so this is
# what "enough to choose from, few enough to solve" works out to: measured,
# TOPTW places about 7.5 sights a day, so a day's shortlist is roughly three
# times what it can use.
SELECTION_PER_DAY = 22
# Hard ceiling on the working set handed to the solver, meals included. Past
# this the solve time stops being interactive: 118 POIs solve in ~2s, a full
# 371-POI catalogue did not solve in ten minutes.
MAX_WORKING_SET = 120


def select_by_score(pois: list[dict], preferences: dict, limit: int,
                    forecast: dict = None) -> list[dict]:
    """The highest-scoring `limit` POIs, best first.

    This is where the score earns its place. Measured against
    `archetype_query` retrieval at equal candidate count, selecting this way
    gave more stops a day at the same distance per stop, better-known places
    and more categories per day -- while using the preference vector directly
    instead of the archetype label the query is built from. Weighting the
    *solver* with the same scores, by contrast, cost stops and distance and
    barely moved preference match at all, so the router selects with the score
    and then optimises geometry uniformly over what it selected. See this
    module's header for the numbers.

    Ties keep the input order, so the same pool always yields the same
    shortlist -- the router's determinism depends on it.
    """
    if limit >= len(pois):
        return list(pois)
    scored = sorted(enumerate(score_pois(pois, preferences, forecast)),
                    key=lambda pair: (-pair[1], pair[0]))
    return [pois[i] for i, _ in scored[:max(limit, 0)]]


if __name__ == "__main__":
    print("preference -> category weights (NNLS):")
    print(pd.DataFrame(PREFERENCE_CATEGORY_WEIGHTS).T.round(3).to_string())
