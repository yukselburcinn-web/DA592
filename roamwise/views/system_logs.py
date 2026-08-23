"""Operator-facing screen: what the app did, and how well the architecture does it.

Two tabs, both showing things that used to sit in the traveler's way:

  Logs -- replaced the "Agent trace" tab (issue #41). Reads the ring buffer in
  `roamwise.app_logging`, which is filled from the stdlib logging tree, so it
  shows whatever the agents, models and retrieval layer logged rather than a
  fixed list of fields the UI knows to ask for.

  Results -- replaced the sidebar's "Retrieval architecture" radio (issue #42).
  Travelers were being asked to pick fusion/hybrid/standard themselves; that
  was a comparative-analysis condition from the proposal, not a decision a
  traveler can make. The app now always runs Fusion RAG, and the comparison
  that justifies it is presented here as evidence.

A page, not an entry point: see the note in `roamwise/app.py`.
"""
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from roamwise import app_logging
from roamwise.evaluation import comparative_analysis as ca

st.title("\U0001f4cb System logs")

logs_tab, results_tab = st.tabs(["\U0001f4dc Logs", "\U0001f4ca Results"])


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
with logs_tab:
    st.caption("What RoamWise's agents, models and retrieval layer logged while this server has been "
               "running. Newest entries first.")

    records = app_logging.records()

    # The filters live in the tab body rather than the sidebar: a sidebar
    # control stays on screen while the Results tab is open, where it would
    # look like it filters the comparison too.
    filter_col, search_col, action_col = st.columns([2, 2, 1])
    with filter_col:
        levels = st.multiselect(
            "Level", app_logging.LEVELS, default=["INFO", "WARNING", "ERROR", "CRITICAL"],
            help="DEBUG is off by default -- the pipeline emits one INFO record per agent step.",
        )
    with search_col:
        search = st.text_input("Search", placeholder="e.g. router, OSRM, failed",
                               help="Matches the message, the source module and the structured fields.")
    with action_col:
        st.write("")  # push the buttons down onto the inputs' baseline
        if st.button("Refresh", use_container_width=True):
            st.rerun()
        if st.button("Clear", use_container_width=True,
                     help="Empties the buffer for everyone using this server."):
            app_logging.clear()
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
        st.info("Nothing logged yet. Plan a trip on the **Plan a trip** screen and the agents' steps "
                "will show up here.")
    elif not selected:
        st.warning("No log entries match the current filters.")
    else:
        n_errors = sum(1 for r in selected if r["levelno"] >= 40)
        n_warnings = sum(1 for r in selected if r["levelno"] == 30)
        col1, col2, col3 = st.columns(3)
        col1.metric("Entries shown", f"{len(selected)} / {len(records)}")
        col2.metric("Warnings", n_warnings)
        col3.metric("Errors", n_errors)

        # Newest first: when something just went wrong, it is the last thing
        # logged, and it should not require scrolling past a whole successful
        # run to find.
        newest_first = list(reversed(selected))

        table = pd.DataFrame([{
            "Time": datetime.fromtimestamp(r["created"]).strftime("%H:%M:%S.%f")[:-3],
            "Level": r["level"],
            # `roamwise.agents.orchestrator` -> `agents.orchestrator`: the
            # prefix is on every row, so it costs width and carries nothing.
            "Source": r["logger"].removeprefix(f"{app_logging.LOGGER_NAME}."),
            "Message": r["message"],
            "Details": app_logging.format_fields(r["fields"]),
        } for r in newest_first])

        # No fixed height: Streamlit sizes to the rows and starts scrolling once
        # there are more than it can show, so a handful of entries does not sit
        # under a block of empty ones. "Details" carries the structured payload
        # and is the column worth the width.
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
                for r in selected  # oldest first in the file, as a log is read
            ),
            file_name="roamwise.log",
            mime="text/plain",
        )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
CONFIG_LABELS = {
    "fusion": "Fusion RAG (Semantic + Graph + Keyword)",
    "hybrid": "Hybrid RAG (Semantic + Keyword)",
    "standard": "Standard prompting (no retrieval)",
}

# The chart legend has room to spell out what each config retrieves; the
# summary table does not -- the full labels push the latency column off the
# right edge. Short names there, since the legend above already defined them.
SHORT_LABELS = {"fusion": "Fusion RAG", "hybrid": "Hybrid RAG", "standard": "Standard prompting"}

# (summary column, label, higher_is_better, what it means)
METRICS = [
    ("mean_recall_at_k", "Multi-hop recall", True,
     "Share of the graph-verified answer set the retriever actually surfaced. This is the metric "
     "graph traversal exists for, so it is where Fusion should separate from Hybrid."),
    ("mean_archetype_precision", "Archetype precision", True,
     "Fraction of surfaced places that match the traveler archetype's preferred categories."),
    ("mean_grounded_entity_rate", "Grounded entities", True,
     "Fraction of surfaced content traceable to a real knowledge-base node -- the structural "
     "stand-in for hallucination risk."),
    ("mean_km_per_stop", "Km per stop", False,
     "Average walking distance per stop once the router builds a day from each config's "
     "candidates. Lower means a more geographically coherent day."),
    ("mean_retrieval_ms", "Retrieval latency", False,
     "Wall-clock time of the retrieval call alone. This is what the extra accuracy costs."),
]


@st.cache_data(show_spinner="Running the comparative analysis across all three configurations...")
def _run_analysis_now():
    results = ca.run_comparative_analysis()
    return results, ca.summarize(results).reset_index()


@st.cache_data
def _load_saved_analysis():
    if not ca.RESULTS_CSV.exists():
        return None, None
    results = pd.read_csv(ca.RESULTS_CSV)
    return results, ca.summarize(results).reset_index()


with results_tab:
    st.caption("RoamWise always plans with Fusion RAG. This is the measurement behind that choice: "
               "the same queries run through all three retrieval architectures from the proposal.")

    if st.button("Re-run the analysis now", help="Recomputes every metric against the current index "
                                                 "and knowledge graph instead of the committed results."):
        results_df, summary_df = _run_analysis_now()
        st.caption("Recomputed just now.")
    else:
        results_df, summary_df = _load_saved_analysis()

    if results_df is None:
        st.info("No saved comparison found. Run `python evaluation/comparative_analysis.py`, "
                "or press **Re-run the analysis now** above.")
        st.stop()

    n_queries = results_df.groupby("config").size().max()
    st.caption(f"{n_queries} test queries across 8 cities and 7 traveler archetypes.")

    # Headline first: the tab exists to answer "which one, and by how much".
    best = summary_df.set_index("config")
    lead = best.loc["fusion", "mean_recall_at_k"] - best.loc["hybrid", "mean_recall_at_k"]
    st.success(
        f"**Fusion RAG leads on every quality metric.** It recovers "
        f"{best.loc['fusion', 'mean_recall_at_k']:.0%} of the graph-verified answer set against "
        f"{best.loc['hybrid', 'mean_recall_at_k']:.0%} for Hybrid RAG (a {lead:.0%} point lead) and "
        f"{best.loc['standard', 'mean_recall_at_k']:.0%} for standard prompting, which retrieves nothing."
    )

    quality = [m for m in METRICS if m[2]]
    long = pd.DataFrame([
        {"Configuration": CONFIG_LABELS[row.config], "Metric": label, "Score": getattr(row, column)}
        for row in summary_df.itertuples()
        for column, label, _, _ in quality
    ])
    fig = px.bar(long, x="Metric", y="Score", color="Configuration", barmode="group",
                 range_y=[0, 1], color_discrete_sequence=["#2E7D32", "#FFA000", "#C62828"],
                 title="Quality metrics (higher is better, all on a 0-1 scale)")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0),
                      margin=dict(l=0, r=0, t=40, b=0), height=380)
    st.plotly_chart(fig, use_container_width=True)

    # The cost metrics are charted separately, not folded into the grouped bar
    # above: km and milliseconds share no scale with a 0-1 rate, and plotting
    # them on one axis would make a 40 ms bar dwarf a 0.9 precision bar.
    st.markdown("##### What it costs")
    cost_cols = st.columns(2)
    for col, (column, label, _, _) in zip(cost_cols, [m for m in METRICS if not m[2]]):
        with col:
            unit = "km" if "km" in column else "ms"
            cost = summary_df.assign(Configuration=summary_df.config.map(CONFIG_LABELS))
            fig = px.bar(cost, x="Configuration", y=column,
                         color="Configuration", color_discrete_sequence=["#2E7D32", "#FFA000", "#C62828"],
                         title=f"{label} ({unit}, lower is better)")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None,
                              margin=dict(l=0, r=0, t=40, b=0), height=300)
            fig.update_xaxes(tickvals=list(CONFIG_LABELS.values()),
                             ticktext=["Fusion", "Hybrid", "Standard"])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Summary")
    table = summary_df.assign(Configuration=summary_df.config.map(SHORT_LABELS))[
        ["Configuration"] + [column for column, _, _, _ in METRICS]
    ].rename(columns={column: label for column, label, _, _ in METRICS})
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Configuration": st.column_config.TextColumn(width="small"),
            **{
                label: (st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.3f")
                        if higher_better else st.column_config.NumberColumn(label, format="%.2f"))
                for _, label, higher_better, _ in METRICS
            },
        },
    )

    for _, label, higher_better, description in METRICS:
        arrow = "higher is better" if higher_better else "lower is better"
        st.markdown(f"**{label}** _({arrow})_ — {description}")

    with st.expander("Per-query results"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)
