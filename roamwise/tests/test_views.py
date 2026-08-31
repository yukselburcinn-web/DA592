"""The sidebar the traveler actually operates: what the preference sliders say
they measure, that renaming one does not repoint what it feeds, and which
orchestrator the picker hands the plan to.

`views/itinerary.py` is the subject here. `test_maps.py` holds the other half
of the same module -- how a plan is drawn -- and the two are kept apart because
they fail for unrelated reasons.
"""

import inspect
import re

import pytest

from roamwise.agents.orchestrator import RoamWiseOrchestrator
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


# --- issue #129: the orchestrator picker ---

def test_the_picker_offers_only_orchestrators_this_install_can_run(monkeypatch):
    """`langgraph` is an optional extra (`requirements-langgraph.txt`), so a
    clean install has to open the app and simply not offer that path. Not
    raise at import, and not offer an option that fails when it is chosen --
    the sidebar is the one place where a missing optional dependency would
    reach a user rather than a developer."""
    monkeypatch.setattr(itinerary, "_langgraph_installed", lambda: False)
    assert list(itinerary._orchestrator_options()) == [RoamWiseOrchestrator.ORCHESTRATOR_ID], \
        "an install without the extra is being offered the LangGraph path"

    monkeypatch.setattr(itinerary, "_langgraph_installed", lambda: True)
    assert list(itinerary._orchestrator_options()) == \
        [RoamWiseOrchestrator.ORCHESTRATOR_ID, "langgraph"]


def test_nothing_at_module_scope_imports_the_optional_extra():
    """The other half of the same guarantee, and the half a monkeypatched
    option list cannot show: `views/itinerary.py` may not import
    `orchestrator_langgraph` at module scope, or the page would fail to load
    on an install without `langgraph` before any picker could hide anything."""
    source = open(itinerary.__file__).read()
    module_scope = [line for line in source.splitlines()
                    if line.startswith(("import ", "from "))]
    assert not [line for line in module_scope if "orchestrator_langgraph" in line], \
        "the optional LangGraph path is imported at module scope"


def test_the_picker_names_the_engines_the_orchestrators_log():
    """The picker's keys are what `get_orchestrator` dispatches on *and* what
    each orchestrator writes as `orchestrator=` on its opening log record,
    which is how the System logs screen names the path that ran. Two spellings
    of the same thing would let the screen and the sidebar disagree."""
    pytest.importorskip("langgraph")
    from roamwise.agents.orchestrator_langgraph import RoamWiseLangGraphOrchestrator

    assert set(itinerary._ENGINE_LABELS) == {RoamWiseOrchestrator.ORCHESTRATOR_ID,
                                             RoamWiseLangGraphOrchestrator.ORCHESTRATOR_ID}


def test_the_chosen_orchestrator_is_part_of_the_engine_cache_key():
    """`get_orchestrator` is `@st.cache_resource`, which keys on the call
    arguments. Reading the picker from session state inside the body instead
    would return the orchestrator built for the *other* engine -- one cached
    object, two labels -- and quietly plan on the path the user just left."""
    params = inspect.signature(itinerary.get_orchestrator).parameters
    assert "engine" in params, \
        "the engine is not an argument, so it cannot be part of the cache key"


def test_the_factory_builds_the_engine_it_is_asked_for(monkeypatch):
    """The dispatch itself, tested off Streamlit's cache -- which is why
    `_build_orchestrator` is a plain function. Constructing either orchestrator
    for real costs a graph build and a model load, so both are stubbed: what is
    under test is which class is reached, not what it does."""
    pytest.importorskip("langgraph")
    from roamwise.agents import orchestrator_langgraph

    monkeypatch.setattr(itinerary, "RoamWiseOrchestrator",
                        lambda retrieval_config: ("custom", retrieval_config))
    monkeypatch.setattr(orchestrator_langgraph, "RoamWiseLangGraphOrchestrator",
                        lambda retrieval_config: ("langgraph", retrieval_config))

    assert itinerary._build_orchestrator("fusion", "custom") == ("custom", "fusion")
    assert itinerary._build_orchestrator("fusion", "langgraph") == ("langgraph", "fusion")
