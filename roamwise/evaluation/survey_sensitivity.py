"""How much of the personalization is a property of `user_survey.csv`? (#124)

Two things in the middle of personalization are calibrated on a **synthetic**
420-row survey that `data/generate_data.py` invented:

  - `optimization/scoring.PREFERENCE_CATEGORY_WEIGHTS` -- the preference ->
    category matrix, fitted by NNLS from the survey against `CATEGORY_AFFINITY`
    (#72). This is the score's preference factor.
  - `models/segmentation.TravelerSegmenter` -- KMeans (k=7) over the survey.
    This decides which archetype a traveler is, and therefore what
    `archetype_query` goes looking for and what hour the day starts at.

REPORT calls the survey "synthetic" in passing and never says how much rests on
it. #124's position is that replacing it with real data is the second question:
measure the dependence first, because the measurement decides whether the
replacement is worth doing at all.

What this measures
------------------
Four perturbations of the survey, each refitting **both** the matrix and the
segmenter, then re-planning the same trips:

  reseed        same archetype centres, same spread, a fresh draw. Isolates
                sampling noise -- the "would a different random survey have
                given a different product?" question in its purest form.
  sigma         the spread of each archetype's cloud, from 0.04 to 0.32
                against the shipped 0.08. Tight clouds are separable; wide
                ones overlap, which is where KMeans should start to fail.
  centre_jitter the archetype centres themselves, moved by +-delta. The
                centres are the 42 hand-written numbers in
                `generate_data.ARCHETYPES`; resampling never perturbs them,
                so nothing else here tests them.
  shuffle       archetype labels permuted across rows. **This is the control,
                not a scenario.** It destroys the correspondence between the
                survey and `CATEGORY_AFFINITY` entirely, so a measurement that
                does not move here cannot see anything, and no "insensitive"
                conclusion may be drawn from the rows above it.

The control also runs decomposed, as `shuffle[matrix]` and `shuffle[segmenter]`,
which install the perturbed fit into one consumer at a time. Without that split
the table is not readable: `centre_jitter` reranks the matrix and moves not one
stop of one plan, and with both consumers perturbed together there is no way to
tell whether the itinerary is insensitive to the matrix or simply dominated by
the archetype label -- which chooses the retrieval query and the hour the day
starts at. Six extra plans' worth of solving buys the answer.

Two rules the numbers depend on
-------------------------------
**Everything is graded on the shipped matrix.** A variant's plans are scored
for preference match with `PREFERENCE_CATEGORY_WEIGHTS` as committed, never
with the variant's own. Letting each variant grade itself is the circularity
#48 removed from `dependence_level`: a collapsed matrix would rate its own
plans perfectly.

**The survey is generated here, never on disk.** `data/generate_data.py` writes
five shipped files in one pass and leaves the app in a half-migrated state that
does not start (CLAUDE.md, and that script's own docstring). Its survey block
is 20 lines -- `generate_data.py:262-281` -- so it is reproduced in
`draw_survey()` below and `data/user_survey.csv` is never touched. The
reproduction is checked against the committed file by
`test_the_resampler_reproduces_the_shipped_survey_s_distribution`.

Run:  python -m roamwise.evaluation.survey_sensitivity
Writes `survey_sensitivity.csv` (one row per variant) and
`survey_sensitivity_plans.csv` (one row per planned trip). Nothing here mutates
a shipped constant beyond the life of a `with` block, and nothing writes to
`data/`.
"""
import contextlib
import sys
import zlib
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from roamwise.agents.orchestrator import RoamWiseOrchestrator
from roamwise.models.segmentation import TravelerSegmenter
from roamwise.optimization import scoring
from roamwise.optimization.scoring import PREFERENCE_DIMS, _fit_preference_matrix

HERE = Path(__file__).parent
SUMMARY_CSV = HERE / "survey_sensitivity.csv"
PLANS_CSV = HERE / "survey_sensitivity_plans.csv"

# The archetype centres and spread `data/generate_data.py:262-281` draws from.
# Copied rather than imported: importing that module executes it, and it
# overwrites five shipped files at import time.
ARCHETYPE_CENTRES = {
    "Culture Enthusiast": dict(budget=0.6, culture=0.95, nature=0.3, nightlife=0.2, relax=0.3, adventure=0.2),
    "Beach & Relax": dict(budget=0.5, culture=0.2, nature=0.6, nightlife=0.3, relax=0.95, adventure=0.2),
    "Budget Backpacker": dict(budget=0.1, culture=0.5, nature=0.5, nightlife=0.6, relax=0.3, adventure=0.6),
    "Luxury Traveler": dict(budget=0.95, culture=0.6, nature=0.3, nightlife=0.4, relax=0.7, adventure=0.2),
    "Nightlife Seeker": dict(budget=0.5, culture=0.3, nature=0.2, nightlife=0.95, relax=0.2, adventure=0.5),
    "Nature & Adventure": dict(budget=0.5, culture=0.2, nature=0.9, nightlife=0.2, relax=0.3, adventure=0.9),
    "Family Traveler": dict(budget=0.6, culture=0.5, nature=0.6, nightlife=0.1, relax=0.6, adventure=0.3),
}
SHIPPED_SIGMA = 0.08
ROWS_PER_ARCHETYPE = 60

# The measurement basis, written down because #126's comment on #124 asks for
# it: whichever of the two measured the orchestrator's pool table first should
# leave its parameters fixed so the second is comparable. #126 closed before
# this ran, so nothing is inherited -- these are the numbers to reuse.
CITIES = ["BER", "PAR"]
N_DAYS = 3
TOP_K_POIS = None          # let the orchestrator scale it (RETRIEVED_POIS_PER_DAY)
DAILY_MINUTES_BUDGET = 480


def draw_survey(rng: np.random.Generator, centres: dict = None,
                sigma: float = SHIPPED_SIGMA,
                rows_per_archetype: int = ROWS_PER_ARCHETYPE) -> pd.DataFrame:
    """One survey, drawn the way `generate_data.py:262-281` draws it.

    Same shape, same clip, same per-archetype row count; only the generator
    object differs (`default_rng` rather than the legacy global `np.random`),
    because this has to produce many independent surveys in one process.
    """
    centres = centres or ARCHETYPE_CENTRES
    rows = []
    for name, centre in centres.items():
        for _ in range(rows_per_archetype):
            rows.append({"archetype": name,
                         **{k: float(np.clip(rng.normal(v, sigma), 0, 1))
                            for k, v in centre.items()}})
    return pd.DataFrame(rows)


def jitter_centres(rng: np.random.Generator, delta: float) -> dict:
    """The hand-written centres, each dimension moved uniformly in +-delta.

    This is the perturbation the issue's framing points at without naming: the
    survey's 420 rows carry only 42 numbers of information into the matrix (see
    `matrix_deltas`), and resampling leaves all 42 where they were.
    """
    return {name: {k: float(np.clip(v + rng.uniform(-delta, delta), 0, 1))
                   for k, v in centre.items()}
            for name, centre in ARCHETYPE_CENTRES.items()}


def shuffle_labels(rng: np.random.Generator, survey: pd.DataFrame) -> pd.DataFrame:
    """Labels permuted across rows, keeping the marginal distribution of every
    feature and every label count exactly as they were. Only the *association*
    between them is destroyed, which is precisely the thing both consumers
    read."""
    out = survey.copy()
    out["archetype"] = rng.permutation(out["archetype"].values)
    return out


@contextlib.contextmanager
def preference_matrix(matrix: dict):
    """Swap the shipped matrix (and the normaliser derived from it) for the
    duration of a block.

    `preference_match` reads both as module globals on every call, so both
    have to move together -- patching the matrix alone would normalise the new
    weights by the old maximum and quietly rescale every score.
    """
    old_matrix, old_max = scoring.PREFERENCE_CATEGORY_WEIGHTS, scoring._MAX_MATCH
    scoring.PREFERENCE_CATEGORY_WEIGHTS = matrix
    scoring._MAX_MATCH = max(sum(matrix[d][c] for d in PREFERENCE_DIMS)
                             for c in next(iter(matrix.values())))
    try:
        yield
    finally:
        scoring.PREFERENCE_CATEGORY_WEIGHTS, scoring._MAX_MATCH = old_matrix, old_max


# ------------------------------------------------------------- the metrics

def matrix_deltas(shipped: dict, variant: dict) -> dict:
    """How far the fitted matrix moved, in weights and in what it ranks first.

    The cell deltas say how much the numbers moved; `top1_agreement` says
    whether the matrix's *ranking* moved, which is a much coarser thing to
    change and the one worth reporting -- a matrix can drift in the third
    decimal everywhere and still rank every category identically.

    Neither column is a statement about the traveler. This measurement's own
    result is that it cannot be: `shuffle[matrix]` reranks five of six
    dimensions and leaves 99.4% of the planned stops in place. What reaches a
    traveler is the archetype label, and that is `archetype_agreement`.
    """
    dims = sorted(shipped)
    cats = sorted(set(shipped[dims[0]]) & set(variant[dims[0]]))
    diff = np.array([[abs(shipped[d][c] - variant[d][c]) for c in cats] for d in dims])
    agree = sum(1 for d in dims
                if max(cats, key=lambda c: shipped[d][c])
                == max(cats, key=lambda c: variant[d][c]))
    return {"matrix_mean_abs_delta": float(diff.mean()),
            "matrix_max_abs_delta": float(diff.max()),
            "matrix_top1_agreement": agree / len(dims)}


def stop_overlap(baseline: list, variant: list) -> float:
    """Jaccard over the POI ids of a whole trip. Two empty plans count as
    identical rather than as a divide-by-zero, which is the honest reading:
    nothing changed."""
    a, b = set(baseline), set(variant)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def plan_preference_match(stops: list[dict], preferences: dict) -> float:
    """Mean preference match of a plan's stops, always under the **shipped**
    matrix. The caller must not have a variant matrix installed here -- see
    the module docstring on self-grading."""
    if not stops:
        return float("nan")
    return float(np.mean([scoring.preference_match(preferences, p.get("category"))
                          for p in stops]))


# ------------------------------------------------------------- the run

def travellers() -> dict[str, dict]:
    """The slider settings measured: one per archetype centre.

    The centres rather than random vectors, because every metric here is a
    comparison against the shipped pipeline and a traveler sitting between two
    archetypes would move for reasons that have nothing to do with the survey.
    """
    return {name: dict(centre) for name, centre in ARCHETYPE_CENTRES.items()}


def variants(seeds: int = 3) -> list[tuple]:
    """(label, survey-builder, channel) triples.

    `channel` says which consumer of the survey the perturbed fit is allowed to
    reach: "both" (what the product would actually do), "matrix" (the NNLS
    weights only, shipped segmenter) or "segmenter" (the KMeans only, shipped
    matrix). Every ordinary perturbation runs on "both"; the control runs on
    all three, because splitting it is the only way to read the table.

    The reason: `centre_jitter` reranks the matrix and moves no plan at all,
    and with both channels perturbed together there is no way to tell whether
    the itinerary is insensitive to the matrix or merely dominated by the
    archetype label, which drives retrieval and the day's start hour. The
    control, decomposed, answers that -- and it costs six plans' worth of
    solving.

    `shuffle` is last so the control reads at the bottom of the table.
    """
    out = [("shipped", None, "both")]
    for s in range(seeds):
        out.append((f"reseed/{s}", lambda rng: draw_survey(rng), "both"))
    for sigma in (0.04, 0.16, 0.32):
        for s in range(seeds):
            out.append((f"sigma={sigma}/{s}",
                        lambda rng, sg=sigma: draw_survey(rng, sigma=sg), "both"))
    for delta in (0.05, 0.15):
        for s in range(seeds):
            out.append((f"centre_jitter={delta}/{s}",
                        lambda rng, d=delta: draw_survey(rng, centres=jitter_centres(rng, d)),
                        "both"))
    for channel in ("both", "matrix", "segmenter"):
        for s in range(seeds):
            label = "shuffle" if channel == "both" else f"shuffle[{channel}]"
            out.append((f"{label}/{s}",
                        lambda rng: shuffle_labels(rng, draw_survey(rng)), channel))
    return out


def run(seeds: int = 3, cities: list = None, n_days: int = N_DAYS) -> tuple:
    """Returns (per-variant summary, per-plan detail)."""
    cities = cities or CITIES
    orch = RoamWiseOrchestrator()
    shipped_matrix = scoring.PREFERENCE_CATEGORY_WEIGHTS
    shipped_segmenter = orch.segmenter
    people = travellers()

    def plan_all(label):
        rows = []
        for city in cities:
            for who, prefs in people.items():
                result = orch.plan_trip(prefs, destination_id=city, n_days=n_days,
                                        daily_minutes_budget=DAILY_MINUTES_BUDGET,
                                        top_k_pois=TOP_K_POIS)
                stops = [p for day in result["routing"]["itinerary"] for p in day["route"]]
                rows.append({"variant": label, "city": city, "traveller": who,
                             "archetype": result["archetype"],
                             "stops": len(stops),
                             "poi_ids": [p.get("poi_id") for p in stops],
                             "categories": [p.get("category") for p in stops],
                             "preferences": prefs})
        return rows

    baseline = plan_all("shipped")
    key = lambda r: (r["city"], r["traveller"])
    base_by = {key(r): r for r in baseline}

    summary, detail = [], []
    for label, build, channel in variants(seeds):
        if build is None:
            matrix, plans = shipped_matrix, baseline
        else:
            # One generator per variant, seeded from the label itself, so a
            # re-run reproduces the table exactly and two variants never share a
            # draw. crc32 rather than hash(): str.__hash__ is salted per
            # process, which would make this reproducible only within one run.
            #
            # The seed is derived from the label, and the decomposed control
            # carries the channel in its label -- so shuffle[matrix]/0 draws a
            # different survey from shuffle/0 rather than re-using it. The three
            # channels are three independent draws of the same perturbation,
            # not one draw split three ways; with 3 seeds each that is the
            # honest reading of the comparison anyway.
            rng = np.random.default_rng(zlib.crc32(label.encode()))
            survey = build(rng)
            matrix = _fit_preference_matrix(survey)
            if channel in ("both", "segmenter"):
                orch.segmenter = TravelerSegmenter(survey=survey)
            with preference_matrix(matrix if channel in ("both", "matrix") else shipped_matrix):
                plans = plan_all(label)
            orch.segmenter = shipped_segmenter

        overlaps, matches, agree = [], [], []
        for row in plans:
            base = base_by[key(row)]
            # Graded on the shipped matrix: `preference_matrix` has exited by
            # now, so this is the committed one whatever the variant fitted.
            row = row | {
                "stop_overlap": stop_overlap(base["poi_ids"], row["poi_ids"]),
                "preference_match": plan_preference_match(
                    [{"category": c} for c in row["categories"]], row["preferences"]),
                "archetype_agrees": row["archetype"] == base["archetype"],
            }
            detail.append({k: ("|".join(str(x) for x in v) if isinstance(v, list) else v)
                           for k, v in row.items() if k != "preferences"})
            overlaps.append(row["stop_overlap"])
            matches.append(row["preference_match"])
            agree.append(row["archetype_agrees"])

        # The matrix columns report the matrix that was actually *in force* for
        # these plans, not the one the variant fitted: on the `segmenter`
        # channel the fitted matrix is discarded, and reporting its delta beside
        # plans it never touched would read as a matrix that moved nothing.
        installed = matrix if (build and channel in ("both", "matrix")) else shipped_matrix
        summary.append({"variant": label, "channel": channel,
                        **matrix_deltas(shipped_matrix, installed),
                        "archetype_agreement": float(np.mean(agree)),
                        "stop_overlap": float(np.mean(overlaps)),
                        "preference_match": float(np.nanmean(matches)),
                        "plans": len(plans)})
        print(f"{label:<24} matrix d={summary[-1]['matrix_mean_abs_delta']:.4f} "
              f"archetype={summary[-1]['archetype_agreement']:.0%} "
              f"stops={summary[-1]['stop_overlap']:.0%} "
              f"pref={summary[-1]['preference_match']:.3f}", flush=True)

    return pd.DataFrame(summary), pd.DataFrame(detail)


def collapse(summary: pd.DataFrame) -> pd.DataFrame:
    """The per-seed rows averaged into one row per perturbation, which is what
    the report quotes."""
    out = summary.copy()
    out["perturbation"] = out.variant.str.split("/").str[0]
    return (out.groupby("perturbation", sort=False)
            .agg(channel=("channel", "first"),
                 seeds=("variant", "size"),
                 matrix_mean_abs_delta=("matrix_mean_abs_delta", "mean"),
                 matrix_max_abs_delta=("matrix_max_abs_delta", "max"),
                 matrix_top1_agreement=("matrix_top1_agreement", "mean"),
                 archetype_agreement=("archetype_agreement", "mean"),
                 stop_overlap=("stop_overlap", "mean"),
                 preference_match=("preference_match", "mean"))
            .reset_index())


if __name__ == "__main__":
    summary, detail = run()
    summary.to_csv(SUMMARY_CSV, index=False)
    detail.to_csv(PLANS_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV.name} and {PLANS_CSV.name}\n")
    print(collapse(summary).round(4).to_string(index=False))
