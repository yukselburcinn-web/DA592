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

# (summary column, per-query column, label, higher_is_better, what it means)
# The per-query column is what the significance test pairs on, so both names
# have to travel together or the two tables cannot be joined.
METRICS = [
    ("mean_recall_at_k", "recall_at_k", "Multi-hop recall", True,
     "Share of the graph-verified answer set the retriever actually surfaced. This is the metric "
     "graph traversal exists for, so it is where Fusion should separate from Hybrid."),
    ("mean_archetype_precision", "archetype_precision", "Archetype precision", True,
     "Fraction of surfaced places that match the traveler archetype's preferred categories."),
    ("mean_grounded_entity_rate", "grounded_entity_rate", "Grounded entities", True,
     "Fraction of surfaced content traceable to a real knowledge-base node -- the structural "
     "stand-in for hallucination risk. It is 1.0 by construction for anything that retrieves at "
     "all, so it separates retrieval from no-retrieval and cannot rank Fusion against Hybrid."),
    ("mean_km_per_stop", "km_per_stop_day1", "Km per stop", False,
     "Average walking distance per stop once the router builds a day from each config's "
     "candidates. Lower means a more geographically coherent day."),
    ("mean_retrieval_ms", "retrieval_ms", "Retrieval latency", False,
     "Wall-clock time of the retrieval call alone. This is what the extra accuracy costs."),
]
METRIC_LABELS = {results_column: label for _, results_column, label, _, _ in METRICS}

VERDICT_TEXT = {
    "better": "Fusion is better",
    "worse": "Fusion is worse",
    "no difference": "No measurable difference",
    "identical": "Identical -- cannot separate them",
}

DEPENDENCE_TEXT = {
    "subset": "Circular — retrieval is a subset of the answer key",
    "superset": "Partly circular — same category filter",
    "independent": "Independent — key never uses the query's routed category",
}

# Compact cell for the breakdown tables: the verdict is what a reader acts on,
# the p-value is what lets them check it, and a full sub-table per slice would
# not fit beside four other metrics.
_VERDICT_SHORT = {"better": "better", "worse": "worse",
                  "no difference": "no difference", "identical": "identical"}


@st.cache_data
def _significance_for(subset: pd.DataFrame) -> pd.DataFrame:
    return ca.paired_significance(subset)


def _verdict_cell(significance: pd.DataFrame, results_column: str) -> str:
    row = significance[(significance.metric == results_column)
                       & (significance.opponent == "hybrid")]
    if row.empty:
        return "—"
    row = row.iloc[0]
    if row.verdict == "identical":
        return "identical"
    return f"{_VERDICT_SHORT[row.verdict]}  (p={row.p_value:.3f})"


@st.cache_data(show_spinner="Running the comparative analysis across all three configurations...")
def _run_analysis_now():
    results = ca.run_comparative_analysis()
    return results, ca.summarize(results).reset_index(), ca.paired_significance(results)


@st.cache_data
def _load_saved_analysis():
    if not ca.RESULTS_CSV.exists():
        return None, None, None
    results = pd.read_csv(ca.RESULTS_CSV)
    return results, ca.summarize(results).reset_index(), ca.paired_significance(results)


with results_tab:
    st.caption("RoamWise always plans with Fusion RAG. This is the measurement behind that choice: "
               "the same queries run through all three retrieval architectures from the proposal.")

    if st.button("Re-run the analysis now", help="Recomputes every metric against the current index "
                                                 "and knowledge graph instead of the committed results."):
        results_df, summary_df, significance_df = _run_analysis_now()
        st.caption("Recomputed just now.")
    else:
        results_df, summary_df, significance_df = _load_saved_analysis()

    if results_df is None:
        st.info("No saved comparison found. Run `python evaluation/comparative_analysis.py`, "
                "or press **Re-run the analysis now** above.")
        st.stop()

    n_queries = results_df.groupby("config").size().max()
    st.caption(f"{n_queries} test queries across 8 cities and 7 traveler archetypes.")

    # The queries themselves, not just their count. This tab exists to justify
    # pinning Fusion RAG, and that argument cannot be checked without reading
    # what was asked and what counted as a correct answer -- the mislabelled
    # answer keys found in #50 were only visible that way (issue #85).
    if "query" in results_df.columns:
        with st.expander(f"Read the {n_queries} queries"):
            st.caption("A query is answered by any place in its accepted categories, narrowed to "
                       "those near a transport hub only when the question asks for that. "
                       "\"Grading\" is how much the answer key leans on the retriever being graded.")
            asked = (results_df.drop_duplicates("query_id")
                     .sort_values(["tier", "query_id"], ascending=[False, True]))
            st.dataframe(
                pd.DataFrame({
                    "Tier": asked.tier,
                    "City": asked.destination_id,
                    "Archetype": asked.archetype,
                    "Query": asked["query"],
                    "Accepted categories": asked.category.str.replace("+", " + ", regex=False),
                    "Near hub": asked.near_transport.map({True: "yes", False: "no"}),
                    "Answers": asked.gold_size,
                    "Grading": asked.dependence.map(DEPENDENCE_TEXT),
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Tier": st.column_config.TextColumn(width="small"),
                    "City": st.column_config.TextColumn(width="small"),
                    "Query": st.column_config.TextColumn(width="large"),
                    "Answers": st.column_config.NumberColumn(
                        help="How many places in the catalogue would answer this query. Only eight "
                             "are retrieved, so a key larger than that caps recall below 1.0."),
                },
            )

    # The headline is derived from the significance test rather than written by
    # hand. The hand-written version claimed Fusion "leads on every quality
    # metric" while it was in fact tied with Hybrid on grounding and
    # indistinguishable on itinerary coherence -- a claim that drifted from the
    # data and stayed wrong (issue #46). Generating it means it cannot.
    vs_hybrid = significance_df[significance_df.opponent == "hybrid"].set_index("metric")
    wins = [METRIC_LABELS[m] for m in vs_hybrid.index if vs_hybrid.loc[m, "verdict"] == "better"]
    draws = [METRIC_LABELS[m] for m in vs_hybrid.index
             if vs_hybrid.loc[m, "verdict"] in ("no difference", "identical")]

    best = summary_df.set_index("config")
    if wins:
        headline = f"**Fusion RAG beats Hybrid RAG on {' and '.join(wins).lower()}** "
        headline += f"(paired Wilcoxon, p < {ca.ALPHA})."
        # Quote the numbers for a metric Fusion actually wins. Quoting a fixed
        # metric meant that once recall became a draw the headline was still
        # citing "28% against 28%" as if it were evidence.
        summary_column = next(column for column, _, label, _, _ in METRICS if label in wins)
        headline += (f" It scores {best.loc['fusion', summary_column]:.2f} on {wins[0].lower()} "
                     f"against {best.loc['hybrid', summary_column]:.2f} for Hybrid and "
                     f"{best.loc['standard', summary_column]:.2f} for standard prompting, which "
                     f"retrieves nothing.")
    else:
        headline = ("**Fusion RAG shows no measurable advantage over Hybrid RAG** on any quality "
                    f"metric (paired Wilcoxon, p >= {ca.ALPHA}).")
    if draws:
        headline += (f" The two are **not distinguishable** on {' and '.join(draws).lower()}, and "
                     f"Fusion is the slower of the two -- see below.")
    st.success(headline)

    quality = [m for m in METRICS if m[3]]
    long = pd.DataFrame([
        {"Configuration": CONFIG_LABELS[row.config], "Metric": label, "Score": getattr(row, column)}
        for row in summary_df.itertuples()
        for column, _, label, _, _ in quality
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
    for col, (column, _, label, _, _) in zip(cost_cols, [m for m in METRICS if not m[3]]):
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
        ["Configuration"] + [column for column, _, _, _, _ in METRICS]
    ].rename(columns={column: label for column, _, label, _, _ in METRICS})
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "Configuration": st.column_config.TextColumn(width="small"),
            **{
                label: (st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="%.3f")
                        if higher_better else st.column_config.NumberColumn(label, format="%.2f"))
                for _, _, label, higher_better, _ in METRICS
            },
        },
    )

    st.markdown("##### Is the lead real?")
    st.caption("A gap between two averages can be noise. Every configuration answers the same "
               "queries, so each metric is tested pairwise with a Wilcoxon signed-rank test. "
               "\"Wins\" counts the queries where Fusion came out ahead on that metric.")
    verdicts = significance_df.assign(
        Metric=significance_df.metric.map(METRIC_LABELS),
        Against=significance_df.opponent.map(SHORT_LABELS),
        Record=significance_df.apply(lambda r: f"{r.wins}W / {r.losses}L / {r.ties}T", axis=1),
        p=significance_df.p_value.map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"),
        Verdict=significance_df.verdict.map(VERDICT_TEXT),
    )[["Metric", "Against", "mean_advantage", "Record", "p", "Verdict"]].rename(
        columns={"mean_advantage": "Mean advantage"})
    st.dataframe(
        verdicts, use_container_width=True, hide_index=True,
        column_config={
            "Mean advantage": st.column_config.NumberColumn(format="%.3f",
                                                            help="Fusion minus the opponent, signed so "
                                                                 "positive always means Fusion is better."),
            "p": st.column_config.TextColumn(width="small"),
        },
    )

    st.markdown("##### Does it replicate?")
    st.caption("The queries come in two tiers: hand-written ones phrased the way a traveler would "
               "actually ask, and a generated grid that sweeps city x category cells evenly. A "
               "claim that only holds in one tier is a property of those queries, not of the "
               "architecture.")
    tiers = [t for t in ("handwritten", "grid") if t in set(results_df.tier)]
    replication = pd.DataFrame([
        {"Metric": label,
         **{f"{tier.capitalize()} (n={results_df[results_df.tier == tier].query_id.nunique()})":
            _verdict_cell(_significance_for(results_df[results_df.tier == tier]), results_column)
            for tier in tiers}}
        for _, results_column, label, _, _ in METRICS
    ])
    st.dataframe(replication, use_container_width=True, hide_index=True)

    st.markdown("##### How much of this rests on circular grading?")
    st.caption("The graph retriever routes on literal keywords. When a query names the answer "
               "key's category next to a transport word it dispatches to the very traversal the "
               "answer key is built from, at a tighter radius -- so its results are a guaranteed "
               "subset of the key. Those queries grade Fusion against itself.")
    dependence = pd.DataFrame([
        {"Grading": DEPENDENCE_TEXT[level],
         "Queries": group.query_id.nunique(),
         **{label: _verdict_cell(_significance_for(group), results_column)
            for _, results_column, label, _, _ in METRICS
            if results_column in ("recall_at_k", "archetype_precision")}}
        for level, group in results_df.groupby("dependence")
    ])
    st.dataframe(dependence, use_container_width=True, hide_index=True)

    for _, results_column, label, higher_better, description in METRICS:
        arrow = "higher is better" if higher_better else "lower is better"
        verdict = vs_hybrid.loc[results_column, "verdict"] if results_column in vs_hybrid.index else None
        flag = ""
        if verdict in ("no difference", "identical"):
            flag = f" &nbsp;`{VERDICT_TEXT[verdict].lower()} vs Hybrid`"
        st.markdown(f"**{label}** _({arrow})_{flag} — {description}")

    with st.expander("Per-query results"):
        st.caption("One row per query and configuration. Every metric above is an average over "
                   "this table.")
        st.dataframe(
            results_df, use_container_width=True, hide_index=True,
            # Without a width the query is squeezed to a few characters between
            # the id columns and the metrics, which is the state this table was
            # already in before it carried the question at all.
            column_config={"query": st.column_config.TextColumn("query", width="large")},
        )
