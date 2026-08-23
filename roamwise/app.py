"""
RoamWise entry point (proposal deliverable #1: "An Interactive Prototype...
allowing a user to input preferences and receive a data-backed, graph-grounded,
optimized itinerary").

This file is only the router. The traveler-facing prototype lives in
`views/itinerary.py`; `views/system_logs.py` is the operator-facing log screen
that replaced the old in-line "Agent trace" tab (issue #41).

Navigation is declared with `st.navigation` rather than by dropping scripts in
a `pages/` directory, because the automatic version labels the entry point
from its *filename* -- travelers would see a nav item called "app". Declaring
the pages lets each one carry a real title.

Run with: streamlit run roamwise/app.py
"""
import sys
from pathlib import Path

# Streamlit puts the *script's* directory (roamwise/) on sys.path, not the repo
# root, so the `roamwise.*` package the rest of the codebase imports would not
# resolve. Put the repo root first so there is exactly one import path for every
# module -- importing the same file under two names loads it twice (see #26).
# This runs before any page does, so the views inherit a fixed sys.path and do
# not repeat the bootstrap.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

from roamwise.app_logging import install as install_log_capture

st.set_page_config(page_title="RoamWise", page_icon="\U0001f9ed", layout="wide",
                   initial_sidebar_state="expanded")

# Route the codebase's logging into the buffer the System logs page reads.
# Idempotent, so re-running this script on every interaction is harmless.
install_log_capture()

_VIEWS = Path(__file__).parent / "views"
st.navigation([
    st.Page(_VIEWS / "itinerary.py", title="Plan a trip", icon="\U0001f9ed", default=True),
    st.Page(_VIEWS / "system_logs.py", title="System logs", icon="\U0001f4cb"),
]).run()
