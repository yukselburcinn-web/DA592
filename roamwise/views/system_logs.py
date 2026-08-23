"""Operator-facing log screen (issue #41).

This is where the diagnostics that used to sit in the traveler's "Agent trace"
tab live now. It reads the ring buffer in `roamwise.app_logging`, which is
filled by the stdlib logging tree -- so it shows whatever the agents, models
and retrieval layer logged, not a fixed list of fields the UI knows to ask for.

A page, not an entry point: see the note in `roamwise/app.py`.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

from roamwise import app_logging

st.title("\U0001f4cb System logs")
st.caption("What RoamWise's agents, models and retrieval layer logged while this server has been running. "
           "Newest entries first.")

records = app_logging.records()

with st.sidebar:
    st.header("Filters")
    levels = st.multiselect(
        "Level", app_logging.LEVELS, default=["INFO", "WARNING", "ERROR", "CRITICAL"],
        help="DEBUG is off by default -- the pipeline emits one INFO record per agent step.",
    )
    search = st.text_input("Search", placeholder="e.g. router, OSRM, failed",
                           help="Matches the message, the source module and the structured fields.")
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        refresh = st.button("Refresh", use_container_width=True)
    with col_b:
        if st.button("Clear", use_container_width=True,
                     help="Empties the buffer for everyone using this server."):
            app_logging.clear()
            st.rerun()
    if refresh:
        st.rerun()

selected = []
for rec in records:
    if rec["level"] not in levels:
        continue
    if search:
        haystack = " ".join([rec["message"], rec["logger"],
                             app_logging.format_fields(rec["fields"])]).lower()
        if search.lower() not in haystack:
            continue
    selected.append(rec)

if not records:
    st.info("Nothing logged yet. Plan a trip on the **RoamWise** screen and the agents' steps will show up here.")
    st.stop()

n_errors = sum(1 for r in selected if r["levelno"] >= 40)
n_warnings = sum(1 for r in selected if r["levelno"] == 30)
col1, col2, col3 = st.columns(3)
col1.metric("Entries shown", f"{len(selected)} / {len(records)}")
col2.metric("Warnings", n_warnings)
col3.metric("Errors", n_errors)

if not selected:
    st.warning("No log entries match the current filters.")
    st.stop()

# Newest first: when something just went wrong, it is the last thing logged,
# and it should not require scrolling past a whole successful run to find.
newest_first = list(reversed(selected))

table = pd.DataFrame([{
    "Time": datetime.fromtimestamp(r["created"]).strftime("%H:%M:%S.%f")[:-3],
    "Level": r["level"],
    # `roamwise.agents.orchestrator` -> `agents.orchestrator`: the prefix is on
    # every single row, so it costs width and carries no information.
    "Source": r["logger"].removeprefix(f"{app_logging.LOGGER_NAME}."),
    "Message": r["message"],
    "Details": app_logging.format_fields(r["fields"]),
} for r in newest_first])

# No fixed height: Streamlit sizes to the rows and starts scrolling once there
# are more than it can show, so a handful of entries does not sit under a block
# of empty ones. "Details" carries the structured payload and is the column
# worth the width, so everything identifying the row is kept narrow.
st.dataframe(
    table, use_container_width=True, hide_index=True,
    column_config={
        "Time": st.column_config.TextColumn(width="small"),
        "Level": st.column_config.TextColumn(width="small"),
        "Source": st.column_config.TextColumn(width="small"),
        "Message": st.column_config.TextColumn(width="medium"),
        "Details": st.column_config.TextColumn(width="large"),
    },
)

tracebacks = [r for r in newest_first if r["traceback"]]
if tracebacks:
    st.subheader("Tracebacks")
    for rec in tracebacks:
        when = datetime.fromtimestamp(rec["created"]).strftime("%H:%M:%S")
        with st.expander(f"{when} · {rec['message']}"):
            st.code(rec["traceback"], language="text")

st.download_button(
    "Download these entries (.log)",
    data="\n".join(
        f"{datetime.fromtimestamp(r['created']).isoformat(timespec='milliseconds')} "
        f"{r['level']:<8} {r['logger']} | {r['message']}"
        + (f" | {app_logging.format_fields(r['fields'])}" if r["fields"] else "")
        + (f"\n{r['traceback']}" if r["traceback"] else "")
        for r in selected  # oldest first in the file, as a log is normally read
    ),
    file_name="roamwise.log",
    mime="text/plain",
)
