"""The sidebar the traveler actually operates: what the preference sliders say
they measure, and that renaming one does not repoint what it feeds.

`views/itinerary.py` is the subject here. `test_maps.py` holds the other half
of the same module -- how a plan is drawn -- and the two are kept apart because
they fail for unrelated reasons.
"""

import re

from roamwise.models.segmentation import FEATURES
from roamwise.optimization.scoring import PREFERENCE_CATEGORY_WEIGHTS
from roamwise.views import itinerary


def _slider_labels() -> list[str]:
    return re.findall(r'_level_slider\(\s*"([^"]+)"', open(itinerary.__file__).read())


def test_every_preference_slider_reads_as_the_same_kind_of_question():
    """Four sliders asked "<what> interest" and two did not, and the two that
    did not were the two nobody could read a direction off.

    "Everyday or upmarket" named both ends and left the level to pick one, so
    Low was either "everyday" or "not very upmarket" depending on the reader.
    "Relaxed pace" sounded like a control over how full a day is -- which is
    what "Time out per day" actually does -- rather than a weight on cafes and
    easy stops, which is what it is."""
    labels = _slider_labels()
    assert len(labels) == 5, f"expected five preference sliders, found {labels}"
    for label in labels:
        assert label.endswith("interest"), \
            f"{label!r} does not read as an interest, so its Low/High has no direction"


def test_renaming_a_slider_did_not_rename_what_it_feeds():
    """#163 is a label fix, not a rename. The keys are the segmentation's
    feature names and the preference matrix's columns; changing them would
    silently repoint the score while looking like a wording change."""
    assert {"budget", "relax"} <= set(FEATURES), \
        "the segmenter's feature names moved with the labels"
    assert {"budget", "relax"} <= set(PREFERENCE_CATEGORY_WEIGHTS), \
        "the preference matrix's rows moved with the labels"


def test_the_budget_slider_is_not_called_a_budget():
    """REPORT §5 states why: `price_level` reaches the free/paid label on a stop
    and the free-entry share in the summary, and neither of those is the score.
    The slider gets there through `shopping` and `landmark` only. Calling it a
    budget or price control would promise filtering the app does not do -- and
    #67 is what an unread price filter already cost once."""
    labels = _slider_labels()
    for label in labels:
        assert not any(word in label.lower() for word in ("budget", "price", "cost")),  \
            f"{label!r} promises the plan is chosen by cost; it is not"
