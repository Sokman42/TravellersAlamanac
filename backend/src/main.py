import joblib
from train import GPMPredictor
import pandas as pd


def main(days=7):
    model_dict = joblib.load("models/gpm_prediction_model.pkl")
    parks = GPMPredictor.get_ontario_parks()
    parks_data = GPMPredictor.collect_historical_for_parks(parks, days_back=365)

    df = GPMPredictor.forecast_next_days_for_parks(parks_data, model_dict, days=days)

    for park_id, group in df.groupby("park_id"):
        park_name = group["park_name"].iloc[0]
        print(f"\n{park_name} ({park_id})")
        for _, r in group.sort_values("date").iterrows():
            print(
                f"  {pd.to_datetime(r['date']).strftime('%Y-%m-%d')}: rain likelihood {r['wet_proba']*100:.0f}%"
            )

    dates = (
        pd.to_datetime(df["date"])
        .drop_duplicates()
        .sort_values()
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )
    print("\nForecast dates from model:", ", ".join(dates))
    if dates:
        print(f"Forecast window: {dates[0]} -> {dates[-1]}")

    today = pd.Timestamp.today().normalize()
    cal_dates = [
        (today + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)
    ]
    print("Calendar dates from today:", ", ".join(cal_dates))
    print(f"Calendar window: {cal_dates[0]} -> {cal_dates[-1]}")

    ranked = GPMPredictor.rank_parks_for_day(df, day_index=1)
    print("\nBest parks for tomorrow (lowest rain risk):")
    for _, r in ranked.iterrows():
        print(f"  {r['park_name']}: rain likelihood {r['wet_proba']*100:.0f}%")


if __name__ == "__main__":
    main(days=7)
