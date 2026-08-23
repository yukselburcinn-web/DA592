"""Emit destinations.csv from common.CITIES.

The repo's version is a hand-written literal in generate_data.py, which means
the city list exists twice and can drift. CITIES already carries every field
the schema needs, so this generates it instead.

`tags` are validated rather than trusted. The orchestrator maps them onto
traveller preferences through a fixed 13-word vocabulary
(agents/orchestrator.py `_tag_affinity`) and silently ignores anything outside
it -- a typo does not raise, it just quietly stops contributing to destination
scoring.

    python build_destinations.py
"""
import json
import sys

import pandas as pd

from common import CITIES, DATA

# agents/orchestrator.py:131 -- tags outside this set are dropped on the floor.
TAG_VOCABULARY = {
    "culture", "history", "art", "beach", "nature", "nightlife", "shopping",
    "food", "religion", "luxury", "budget", "romance", "music",
}

COLUMNS = ["destination_id", "city", "country", "lat", "lon", "budget_level", "tags"]


def main():
    rows, problems = [], []
    for code, c in CITIES.items():
        unknown = set(c["tags"]) - TAG_VOCABULARY
        if unknown:
            problems.append(f"{code}: sozlukte olmayan etiket {sorted(unknown)}")
        if not 1 <= c["budget_level"] <= 3:
            problems.append(f"{code}: budget_level {c['budget_level']} 1-3 disinda")
        rows.append({
            "destination_id": code,
            "city": c["city"],
            "country": c["country"],
            "lat": c["lat"],
            "lon": c["lon"],
            "budget_level": c["budget_level"],
            "tags": json.dumps(c["tags"]),
        })

    if problems:
        sys.exit("HATA:\n  " + "\n  ".join(problems))

    df = pd.DataFrame(rows)[COLUMNS]
    out = DATA / "destinations.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}  ({len(df)} satir, etiketler dogrulandi)")


if __name__ == "__main__":
    main()
