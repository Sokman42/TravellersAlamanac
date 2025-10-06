from train import GPMPredictor

parks = GPMPredictor.get_ontario_parks()
parks_data = GPMPredictor.collect_historical_for_parks(parks, days_back=365)

pred = GPMPredictor()
X, y = GPMPredictor.build_training_set(parks_data, pred)
model_dict, metrics = pred.train_model(X, y)

ui_payload = GPMPredictor.forecast_for_ui(parks_data, model_dict, days=14)

fcst_df = GPMPredictor.forecast_next_days_for_parks(parks_data, model_dict, days=14)
tomorrow_ranked = GPMPredictor.rank_parks_for_day(fcst_df, day_index=1)

top_list = tomorrow_ranked.to_dict(orient="records")
