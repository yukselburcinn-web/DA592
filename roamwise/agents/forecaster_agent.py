"""Forecaster Agent: interprets the Holt-Winters demand model's outputs and
turns them into a destination recommendation / crowding narrative."""
from roamwise.agents.llm_client import LLMClient, get_default_llm_client
from roamwise.models.forecasting import forecast_city, load_demand


class ForecasterAgent:
    def __init__(self, llm: LLMClient = None):
        self.llm = llm or get_default_llm_client()
        self._demand = load_demand()

    def run(self, destination_id: str, travel_month: str = None, horizon_months: int = 12) -> dict:
        """travel_month: 'YYYY-MM' if the user already picked a month; otherwise
        the agent recommends the lowest-crowding month in the horizon."""
        fc = forecast_city(destination_id, horizon_months=horizon_months, df=self._demand)

        if travel_month:
            row = fc[fc.date.dt.strftime("%Y-%m") == travel_month]
            target = row.iloc[0] if len(row) else fc.iloc[0]
        else:
            target = fc.sort_values("crowding_z").iloc[0]

        best_alternatives = fc.sort_values("crowding_z").head(3)
        narrative = self._narrate(destination_id, target, best_alternatives)

        return {
            "destination_id": destination_id,
            "target_month": target.date.strftime("%Y-%m"),
            "forecast_visitors": int(target.forecast_visitors),
            "crowding_level": str(target.crowding_level),
            "low_crowd_alternatives": [
                {"month": r.date.strftime("%Y-%m"), "crowding_level": str(r.crowding_level)}
                for r in best_alternatives.itertuples()
            ],
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


if __name__ == "__main__":
    agent = ForecasterAgent()
    result = agent.run("BCN")
    print(result["narrative"])
    print(result)
