"""Do the query phrasings match the words the corpus uses -- and what does
fixing them cost? (issue #123)

#113 established a principle for one category: a query phrase has to carry a
word the documents actually contain, because `retrieval/corpus.py::tokenize`
does no stemming and BM25 can only match what is literally there. `religion`
asked for `worship` (4 of 654 documents) while the documents said `church`
(69), and the category was effectively invisible to keyword retrieval.

Applying that principle to every entry turned up two more in the same state:

    landmark   `landmarks`   2 documents   (`landmark`: 117)
    museum     `museums`    17 documents   (`museum`:  127)

They went unnoticed for the reason `religion` did not: the graph retriever
carries both at 0.9 and 1.0 affinity, so something always surfaced.

**Why this is a sweep and not a one-line fix.** The retrieval pool is strictly
zero-sum at a fixed `top_k` -- 72 POIs is 72 POIs. Strengthening one category's
lexical match pushes another out of the fused top-72, and measured on the
shipped data the category it pushes out is `religion`: from the 30 reachable
POIs #113 won back down to 23. So phrasing cannot be tuned on its own. The
second knob is `build_graph.PREFERENCE_QUOTA_EXPONENT`, which decides how
sharply the graph retriever's ranking favours an archetype's strongest
categories, and the two are swept together here.

What it reports, per configuration:

  total        how many of the 654 POIs any traveler could be shown, unioned
               over 2 cities x 7 archetypes -- `retrieval_coverage.py`'s
               measure, at the three-day pool the app actually retrieves
  per category the same, split out, because the fault this exists for was
               invisible in the total
  iconic       of each city's twelve best-known POIs, how many are reachable.
               Total can be gamed by trading famous places for obscure ones;
               this is what catches it.
  losses       which categories end up below where they are today. Printed
               rather than summed, because "+16 and -3" is a trade someone has
               to agree to, not a number that decides itself.

Run:  python -m roamwise.evaluation.category_phrase_sweep
"""
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import itertools

import pandas as pd

from roamwise.agents.fusion_rag_agent import FusionRAGAgent
from roamwise.agents.orchestrator import RETRIEVED_POIS_PER_DAY
from roamwise.knowledge_graph import build_graph
from roamwise.knowledge_graph.build_graph import CATEGORY_AFFINITY, DATA_DIR
from roamwise.retrieval import query as query_module

HERE = Path(__file__).parent
SWEEP_CSV = HERE / "category_phrase_sweep.csv"

CITIES = ["PAR", "BER"]
N_DAYS = 3
ICONIC_PER_CITY = 12

# The candidates, and why these. Each replacement carries the singular form the
# corpus actually uses; the `museums`/`landmarks` rows are the current state,
# kept in the grid so every table has its own baseline in it rather than in a
# comment somebody has to trust.
MUSEUM_PHRASES = ["museums", "museum collections", "museum exhibitions",
                  "museums and museum collections", "museum art collections"]
LANDMARK_PHRASES = ["landmarks", "landmark sights", "landmark monuments",
                    "landmarks and landmark monuments"]
# `religion` is in the grid because it is the category the other two break, not
# because it was broken -- #113 had already fixed it. The candidates name the
# specific buildings the corpus names (`cathedral` 15, `basilica` 7, `chapel` 9)
# rather than the abstraction it does not (`worship` 4, `places` 5).
RELIGION_PHRASES = ["church buildings and places of worship",
                    "church buildings cathedrals and chapels",
                    "church cathedral basilica and chapel buildings"]
# 1.0 is proportional to the archetype's stated preference; below it flattens
# towards equal shares. Anything above 1.0 concentrates the cut further on the
# strongest categories, which is the direction #113 spent an issue undoing.
QUOTA_EXPONENTS = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]

# Stage two measures every trip length rather than only the default. That is
# not thoroughness for its own sake: the configuration this issue would
# otherwise have shipped looked clean at three days (`religion` held at 30) and
# was quietly costing it 8 POIs at five. A fix measured only where the app
# usually runs is a fix that hides wherever it does not.
TRIP_LENGTHS = [1, 3, 5]


def reachable(rag: FusionRAGAgent, top_k: int) -> set:
    """Every POI some traveler could be shown -- the union over archetypes and
    cities. `archetype_query` reads only the archetype, so this is the whole
    space rather than a sample of it (`retrieval_coverage.py`)."""
    found = set()
    for city in CITIES:
        for archetype in sorted(CATEGORY_AFFINITY):
            out = rag.run(query_module.archetype_query(archetype), destination_id=city,
                          archetype=archetype, config="fusion", top_k=top_k, narrate=False)
            found |= {r["poi_id"] for r in out["results"] if r.get("type") == "poi"}
    return found


def main():
    rag = FusionRAGAgent()
    poi = pd.read_csv(DATA_DIR / "poi.csv")
    iconic = pd.concat([poi[poi.destination_id == c].nlargest(ICONIC_PER_CITY,
                                                              "popularity_score")
                        for c in CITIES])
    categories = sorted(poi.category.unique())
    shipped_phrases = dict(query_module.CATEGORY_PHRASE)
    shipped_exponent = build_graph.PREFERENCE_QUOTA_EXPONENT

    def measure(top_k):
        found = reachable(rag, top_k)
        held = poi.assign(hit=poi.poi_id.isin(found))
        per = held.groupby("category").hit.sum().to_dict()
        return (len(found), {c: int(per.get(c, 0)) for c in categories},
                int(iconic.poi_id.isin(found).sum()))

    # The baseline every row is read against: today's phrasings at a
    # proportional quota, i.e. the state before this issue -- measured at every
    # trip length, because a row can only be called a regression against the
    # same trip length.
    query_module.CATEGORY_PHRASE = dict(shipped_phrases, museum="museums",
                                        landmark="landmarks",
                                        religion="church buildings and places of worship")
    build_graph.PREFERENCE_QUOTA_EXPONENT = 1.0
    baselines = {n: measure(n * RETRIEVED_POIS_PER_DAY) for n in TRIP_LENGTHS}
    base_total, base_per, base_iconic = baselines[N_DAYS]
    for n_days, (total, per, hit) in baselines.items():
        print(f"baseline {n_days}-day: {total} reachable, iconic {hit}/{len(iconic)}, "
              + " ".join(f"{c[:3]}={per[c]}" for c in categories), flush=True)

    rows = []
    try:
        # Stage one: what the phrasings do, and that the quota alone cannot
        # undo it. Three-day pool, which is what the app retrieves by default.
        print("\nstage 1 -- phrasing x quota, 3-day pool", flush=True)
        for exponent, museum, landmark in itertools.product(
                [1.0, 0.5], MUSEUM_PHRASES, LANDMARK_PHRASES):
            build_graph.PREFERENCE_QUOTA_EXPONENT = exponent
            query_module.CATEGORY_PHRASE = dict(shipped_phrases, museum=museum,
                                                landmark=landmark,
                                                religion=RELIGION_PHRASES[0])
            total, per, iconic_hit = measure(N_DAYS * RETRIEVED_POIS_PER_DAY)
            losses = {c: base_per[c] - per[c] for c in categories if per[c] < base_per[c]}
            rows.append({
                "stage": 1, "n_days": N_DAYS, "quota_exponent": exponent,
                "museum_phrase": museum, "landmark_phrase": landmark,
                "religion_phrase": RELIGION_PHRASES[0],
                "total_reachable": total, "iconic_reachable": iconic_hit,
                **{f"reachable_{c}": per[c] for c in categories},
                "categories_below_baseline": len(losses),
                "losses": "; ".join(f"{c} -{n}" for c, n in sorted(losses.items())) or "-",
            })
            print(f"  exp={exponent}  {museum:30s} {landmark:32s} "
                  f"total {total:4d}  iconic {iconic_hit:2d}  "
                  f"religion {per.get('religion', 0):3d}  "
                  f"{rows[-1]['losses']}", flush=True)

        # Stage two: with the phrasings fixed, what protects `religion` -- and
        # at which trip lengths. This is where the shipped exponent comes from.
        print("\nstage 2 -- religion phrasing x quota, every trip length", flush=True)
        for exponent, religion in itertools.product(QUOTA_EXPONENTS, RELIGION_PHRASES):
            build_graph.PREFERENCE_QUOTA_EXPONENT = exponent
            query_module.CATEGORY_PHRASE = dict(
                shipped_phrases, museum=shipped_phrases["museum"],
                landmark=shipped_phrases["landmark"], religion=religion)
            cells = []
            for n_days in TRIP_LENGTHS:
                total, per, iconic_hit = measure(n_days * RETRIEVED_POIS_PER_DAY)
                losses = {c: baselines[n_days][1][c] - per[c]
                          for c in categories if per[c] < baselines[n_days][1][c]}
                rows.append({
                    "stage": 2, "n_days": n_days, "quota_exponent": exponent,
                    "museum_phrase": shipped_phrases["museum"],
                    "landmark_phrase": shipped_phrases["landmark"],
                    "religion_phrase": religion,
                    "total_reachable": total, "iconic_reachable": iconic_hit,
                    **{f"reachable_{c}": per[c] for c in categories},
                    "categories_below_baseline": len(losses),
                    "losses": "; ".join(f"{c} -{n}" for c, n in sorted(losses.items())) or "-",
                })
                cells.append(f"{n_days}d {total:4d} rel {per.get('religion', 0):3d}"
                             f"{'*' if 'religion' in losses else ' '}")
            print(f"  exp={exponent}  {religion:46s} " + "  ".join(cells), flush=True)
    finally:
        query_module.CATEGORY_PHRASE = shipped_phrases
        build_graph.PREFERENCE_QUOTA_EXPONENT = shipped_exponent

    df = pd.DataFrame(rows)
    df.to_csv(SWEEP_CSV, index=False)
    print(f"\nwrote {SWEEP_CSV}")

    # The shipped configuration, named rather than left to be inferred from the
    # table -- a sweep whose conclusion is not written down is a table.
    chosen = df[(df.quota_exponent == shipped_exponent)
                & (df.museum_phrase == shipped_phrases["museum"])
                & (df.landmark_phrase == shipped_phrases["landmark"])
                & (df.religion_phrase == shipped_phrases["religion"])]
    print("\nshipped configuration:")
    print(chosen.to_string(index=False) if not chosen.empty
          else "  (not in the grid -- the shipped phrasings moved without a sweep)")


if __name__ == "__main__":
    main()
