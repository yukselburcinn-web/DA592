"""Forecaster Agent: interprets the Holt-Winters demand model's outputs and
turns them into a destination recommendation / crowding narrative."""
import pandas as pd

from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.models.forecasting import forecast_city, load_demand

MAX_HORIZON_MONTHS = 36


class ForecasterAgent:
    def __init__(self, llm: LLMClient = None):
        self.llm = llm or get_default_llm_client()
        self._demand = load_demand()

    def _horizon_for(self, destination_id: str, travel_month: str,
                     horizon_months: int) -> int:
        """Forecast far enough ahead to actually reach the month being asked
        about.

        forecast_city() counts its horizon from the end of that city's history,
        not from today, and the two are not the same point: Eurostat publishes
        German regional tourism about 20 months in arrears, so Berlin's history
        stops well before Paris's and a 12-month horizon from it lands in the
        past. Asked about a month outside that window, this agent used to fall
        back to the first forecast row without saying so -- a question about
        July came back with the previous January, which is a low-crowding month
        in Berlin where July is high. _recommend_destination turns crowding
        straight into a score penalty, so the stale city won every summer
        comparison it was in.
        """
        history = self._demand[self._demand.destination_id == destination_id]
        if history.empty:
            return horizon_months
        end = pd.Period(history.date.max(), freq="M")

        target = (pd.Period(travel_month, freq="M") if travel_month
                  else pd.Period(pd.Timestamp.today(), freq="M") + horizon_months)
        needed = (target - end).n
        return max(horizon_months, min(needed, MAX_HORIZON_MONTHS))

    @staticmethod
    def _future(fc):
        """Forecast rows from the current month onward, or the whole frame if
        every row is already in the past (better a stale answer than none)."""
        ahead = fc[fc.date >= pd.Timestamp.today().normalize().replace(day=1)]
        return ahead if len(ahead) else fc

    def run(self, destination_id: str, travel_month: str = None, horizon_months: int = 12,
            narrate: bool = True) -> dict:
        """travel_month: 'YYYY-MM' if the user already picked a month; otherwise
        the agent recommends the lowest-crowding month in the horizon."""
        horizon = self._horizon_for(destination_id, travel_month, horizon_months)
        fc = forecast_city(destination_id, horizon_months=horizon, df=self._demand)

        requested_month_available = True
        if travel_month:
            row = fc[fc.date.dt.strftime("%Y-%m") == travel_month]
            # Still possible past MAX_HORIZON_MONTHS. Fall back to the same
            # calendar month rather than the first row, so at least the season
            # is right, and report that it happened instead of hiding it.
            if len(row):
                target = row.iloc[0]
            else:
                requested_month_available = False
                month_no = pd.Period(travel_month, freq="M").month
                same_month = fc[fc.date.dt.month == month_no]
                target = (same_month.iloc[-1] if len(same_month) else fc.iloc[0])
        else:
            target = self._future(fc).sort_values("crowding_z").iloc[0]

        # Recommendations have to be months the traveller can still book. The
        # forecast now starts at the end of each city's history, and for a city
        # whose source lags that start is in the past -- so the cheapest month
        # in the whole horizon was one that had already been and gone.
        best_alternatives = self._future(fc).sort_values("crowding_z").head(3)
        # narrate=False for the callers that only want a number. Destination
        # selection scores every city in the catalogue through this method and
        # reads `crowding_level` alone, so narrating there spent one full
        # generation per candidate city and threw the prose away -- N+1
        # generations per request, of which N were never shown to anyone.
        # #57 gave FusionRAGAgent and RouterAgent this same flag; the
        # forecaster was missed, which is where the cost came back (issue #125).
        narrative = self._narrate(destination_id, target, best_alternatives) if narrate else None

        history = self._demand[self._demand.destination_id == destination_id]
        return {
            "destination_id": destination_id,
            "target_month": target.date.strftime("%Y-%m"),
            "forecast_visitors": int(target.forecast_visitors),
            "crowding_level": str(target.crowding_level),
            "low_crowd_alternatives": [
                {"month": r.date.strftime("%Y-%m"), "crowding_level": str(r.crowding_level)}
                for r in best_alternatives.itertuples()
            ],
            # Staleness is reported, not hidden: source coverage differs per
            # city and a caller comparing two of them should be able to see it.
            "data_through": str(history.date.max())[:7] if not history.empty else None,
            "horizon_months_used": horizon,
            "requested_month_available": requested_month_available,
            "narrative": narrative,
        }

    def _narrate(self, destination_id: str, target, alternatives) -> str:
        alt_text = ", ".join(f"{r.date.strftime('%B %Y')} ({r.crowding_level})" for r in alternatives.itertuples())
        prompt = f"""
        Forecast for {destination_id} in {target.date.strftime('%B %Y')}: expected demand is
        {target.crowding_level} (~{int(target.forecast_visitors):,} monthly visitors, vs this
        city's typical range). If your dates are flexible, the lowest-crowding months in the
        next {12} months are: {alt_text}.
        """
        return self.llm.complete(system="You are a concise travel-demand analyst.", prompt=prompt)


# Demo blocks below take their city from the catalogue rather than naming one:
# a hardcoded code prints nothing at all once that city stops shipping.
def _demo_city():
    import pandas as pd
    from pathlib import Path as _P
    d = _P(__file__).resolve().parents[1] / "data" / "destinations.csv"
    return pd.read_csv(d).destination_id.iloc[0]


if __name__ == "__main__":
    agent = ForecasterAgent()
    result = agent.run(_demo_city())
    print(result["narrative"])
    print(result)
