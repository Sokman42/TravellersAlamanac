from netCDF4 import Dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
from requests.auth import HTTPBasicAuth
import netrc
import os
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
import joblib


class GPMPredictor:
    def __init__(self, lat=43.7, lon=-79.4):
        """
        Initialize GPM precipitation predictor

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
        """
        self.lat = lat
        self.lon = lon
        # Normalize input longitude to 0–360 convention used by IMERG longitude grids.
        # See: NASA GPM IMERG V07 file specification (lon in degrees_east often 0..360)
        # and CF Conventions for longitude ranges (either -180..180 or 0..360).
        self.lon_360 = lon + 360 if lon < 0 else lon
        self.data_cache = "data/gpm_historical_data.pkl"
        self.model_cache = "models/gpm_prediction_model.pkl"

        # Setup authentication
        try:
            netrc_info = netrc.netrc()
            auth = netrc_info.authenticators("urs.earthdata.nasa.gov")
            if not auth:
                raise Exception("No credentials for urs.earthdata.nasa.gov in .netrc")
            self.username, _, self.password = auth
        except Exception as e:
            raise Exception(f"Error reading .netrc: {e}")

    def download_historical_data(self, start_date, end_date):
        """
        Download historical GPM data for training

        Args:
            start_date: datetime object for start
            end_date: datetime object for end

        Returns:
            pandas.DataFrame with date and precipitation columns
        """
        print(
            f"Downloading historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        data = []
        current_date = start_date

        session = requests.Session()
        session.auth = HTTPBasicAuth(self.username, self.password)
        session.headers.update(
            {"User-Agent": "python-requests/2.28.1 NASA GES DISC Data Access"}
        )

        while current_date <= end_date:
            try:
                # Format date for URL
                year = current_date.strftime("%Y")
                month = current_date.strftime("%m")
                day_str = current_date.strftime("%Y%m%d")

                # Construct download URL (using /data which works reliably)
                url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07/{year}/{month}/3B-DAY-L.MS.MRG.3IMERG.{day_str}-S000000-E235959.V07B.nc4"

                head = session.head(url, timeout=10)

                if head.status_code != 200:
                    print(
                        f"Skipping {current_date:%Y-%m-%d} (Status: {head.status_code})"
                    )
                    current_date += timedelta(days=1)
                    continue

                print(f"Processing {current_date:%Y-%m-%d}...", end=" ")
                temp_file = f"temp_{day_str}.nc4"
                precip = None
                downloaded = False
                if not os.path.exists(temp_file):
                    get_resp = session.get(url, stream=True, timeout=60)
                    if get_resp.status_code != 200:
                        print(f"GET failed ({get_resp.status_code})")
                        current_date += timedelta(days=1)
                        continue
                    with open(temp_file, "wb") as f:
                        for chunk in get_resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    downloaded = True
                print(
                    f"Downloaded {temp_file}, size: {os.path.getsize(temp_file)} bytes"
                )
                try:
                    with Dataset(temp_file) as ds:
                        lats = ds.variables["lat"][:]
                        lons = ds.variables["lon"][:]
                        lat_idx = int(np.abs(lats - self.lat).argmin())
                        target_lon = (
                            self.lon_360 if float(np.nanmax(lons)) > 180 else self.lon
                        )
                        lon_idx = int(np.abs(lons - target_lon).argmin())

                        # Select precip var
                        if "precipitationCal" in ds.variables:
                            var_name = "precipitationCal"
                        elif "precipitation" in ds.variables:
                            var_name = "precipitation"
                        else:
                            var_name = next(
                                (
                                    k
                                    for k in ds.variables.keys()
                                    if "precip" in k.lower()
                                ),
                                None,
                            )
                            if var_name is None:
                                raise KeyError(
                                    f"No precipitation var found. Available: {list(ds.variables.keys())[:10]}"
                                )
                        # Read value at (time, lat, lon)
                        val = ds.variables[var_name][0, lat_idx, lon_idx]
                        # Handle masked arrays and NaNs gracefully
                        val = np.ma.filled(val, np.nan)
                        precip = float(np.nan_to_num(val, nan=0.0))
                finally:
                    if downloaded:
                        try:
                            os.remove(temp_file)
                        except Exception:
                            pass
                data.append(
                    {
                        "date": current_date,
                        "precipitation": precip,
                        "lat": float(self.lat),
                        "lon": float(self.lon),
                        "year": current_date.year,
                        "month": current_date.month,
                        "day": current_date.day,
                        "day_of_year": current_date.timetuple().tm_yday,
                        "weekday": current_date.weekday(),
                    }
                )
                amount_str = f"{precip:.2f}mm" if precip is not None else "n/a"
                print(f"✓ Precipitation: {amount_str}")
            except Exception as e:
                print(f"Error {current_date:%Y-%m-%d}: {str(e)[:80]}")

            current_date += timedelta(days=1)
        df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)
        joblib.dump(df, self.data_cache)
        print(f"\n✓ Collected {len(df)} days of data and saved to {self.data_cache}")
        return df

    def load_or_download_data(self, days_back=365):
        """
        Load cached data or download if not available

        Args:
            days_back: How many days of historical data to collect

        Returns:
            pandas.DataFrame with historical data
        """
        if os.path.exists(self.data_cache):
            print("Loading cached historical data...")
            df = joblib.load(self.data_cache)
            print(f"Loaded {len(df)} days of cached data")
            return df
        else:
            print("No cached data found. Downloading historical data...")
            end_date = datetime.now() - timedelta(
                days=3
            )  # Account for processing delay
            start_date = end_date - timedelta(days=days_back)
            return self.download_historical_data(start_date, end_date)

    def create_features(self, df):
        """
        Create features for machine learning model

        Args:
            df: DataFrame with historical data

        Returns:
            X (features), y (target precipitation)
        """
        # Create lag features (previous days' precipitation)
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"precip_lag_{lag}"] = df["precipitation"].shift(lag)

        # Rolling stats on past-only values (shift(1) prevents leakage)
        past = df["precipitation"].shift(1)
        for window in [7, 14, 30]:
            df[f"precip_mean_{window}d"] = past.rolling(
                window=window, min_periods=window
            ).mean()
            df[f"precip_std_{window}d"] = past.rolling(
                window=window, min_periods=window
            ).std()
            df[f"precip_max_{window}d"] = past.rolling(
                window=window, min_periods=window
            ).max()
            df[f"precip_sum_{window}d"] = past.rolling(
                window=window, min_periods=window
            ).sum()

        # Simple seasonality (helps a lot for precip)
        if "day_of_year" not in df.columns:
            df["day_of_year"] = df["date"].dt.dayofyear
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Wet flags from yesterday (no leakage)
        df["prev_wet"] = (df["precip_lag_1"] > 0.1).astype(int)  # 0.1 mm threshold
        # Days since last rain (computed from prev_wet)
        g = df["prev_wet"].cumsum()
        df["days_since_rain"] = g.groupby(g).cumcount()

        # Precipitation intensity category from yesterday's precip (no leakage)
        df["precip_intensity_prev"] = pd.cut(
            past,
            bins=[-np.inf, 0.1, 1.0, 5.0, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)

        # Monthly climatology from prior values only (no leakage)
        df["monthly_climatology"] = df.groupby("month")["precipitation"].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        df["seasonal_anomaly"] = past - df["monthly_climatology"]

        # Recent trend (7-day slope) on past values only
        def slope7(arr):
            x = np.arange(len(arr))
            return np.polyfit(x, np.asarray(arr, dtype=float), 1)[0]

        df["precip_trend_7d"] = past.rolling(7, min_periods=7).apply(slope7, raw=True)

        # Weekend effect
        if "weekday" not in df.columns:
            df["weekday"] = df["date"].dt.weekday
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        # Finalize dataset
        df = df.dropna().reset_index(drop=True)
        feature_cols = [
            c
            for c in df.columns
            if c not in ["date", "precipitation", "park_id", "park_name"]
        ]
        X = df[feature_cols]
        y = df["precipitation"]
        return X, y, df

    def train_model(self, X, y, wet_thresh=0.1):
        """
        Two-stage model: wet/dry classifier + amount regressor (wet only).
        Returns: model_dict, metrics
        """
        n = len(X)
        if n < 90:
            raise ValueError("Not enough data to train (need ~90+ days).")

        # Time-based split
        split_idx = int(n * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        # Labels for classifier
        y_train_wet = (y_train > wet_thresh).astype(int)
        y_val_wet = (y_val > wet_thresh).astype(int)

        # Handle class imbalance
        pos = y_train_wet.sum()
        neg = len(y_train_wet) - pos
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0

        # Stage 1: wet/dry classifier
        clf = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        clf.fit(
            X_train,
            y_train_wet,
            eval_set=[(X_val, y_val_wet)],
            verbose=False,
        )

        clf_cal = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
        clf_cal.fit(X_val, y_val_wet)

        val_proba = clf.predict_proba(X_val)[:, 1]

        # Choose threshold to maximize F1 on validation
        precisions, recalls, thresholds = precision_recall_curve(y_val_wet, val_proba)
        f1s = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
        best_idx = int(np.nanargmax(f1s))
        best_thresh = (
            float(thresholds[max(best_idx - 1, 0)])
            if best_idx < len(thresholds)
            else 0.5
        )

        # Stage 2: regressor on wet-only
        wet_mask_train = y_train > wet_thresh
        reg = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
        )
        if wet_mask_train.any():
            reg.fit(
                X_train[wet_mask_train],
                y_train[wet_mask_train],
                eval_set=[(X_val[y_val > wet_thresh], y_val[y_val > wet_thresh])],
                verbose=False,
            )
        else:
            warnings.warn("No wet days in training; regressor not trained.")

        # Combined prediction on validation
        pred_is_wet = val_proba >= best_thresh
        reg_preds = np.zeros(len(X_val), dtype=float)
        if hasattr(reg, "predict") and (y_val > wet_thresh).any():
            try:
                reg_preds[pred_is_wet] = np.clip(
                    reg.predict(X_val[pred_is_wet]), 0, None
                )
            except Exception:
                pass
        combined = reg_preds  # 0 for predicted dry, amount for predicted wet

        # Metrics
        cls_prec = precision_score(y_val_wet, pred_is_wet, zero_division=0)
        cls_rec = recall_score(y_val_wet, pred_is_wet, zero_division=0)
        cls_f1 = f1_score(y_val_wet, pred_is_wet, zero_division=0)
        try:
            cls_auc = roc_auc_score(y_val_wet, val_proba)
        except Exception:
            cls_auc = np.nan

        overall_mae = mean_absolute_error(y_val, combined)
        overall_rmse = np.sqrt(mean_absolute_error(y_val, combined))
        wet_mask_val = y_val > wet_thresh
        wet_mae = (
            mean_absolute_error(y_val[wet_mask_val], combined[wet_mask_val])
            if wet_mask_val.any()
            else np.nan
        )

        metrics = {
            "clf_precision": float(cls_prec),
            "clf_recall": float(cls_rec),
            "clf_f1": float(cls_f1),
            "clf_auc": float(cls_auc),
            "mae": float(overall_mae),
            "rmse": float(overall_rmse),
            "wet_mae": float(wet_mae),
            "threshold": best_thresh,
        }

        # Save combined model
        model_dict = {
            "classifier": clf_cal,
            "regressor": reg,
            "wet_threshold": wet_thresh,
            "decision_threshold": best_thresh,
            "feature_names": list(X.columns),
        }
        os.makedirs(os.path.dirname(self.model_cache), exist_ok=True)
        joblib.dump(model_dict, self.model_cache)
        print(
            f"Cls F1: {cls_f1:.3f} (P:{cls_prec:.3f}, R:{cls_rec:.3f}, AUC:{cls_auc:.3f}) | "
            f"MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}, Wet MAE: {wet_mae:.3f}, thr={best_thresh:.2f}"
        )
        return model_dict, metrics

    def predict(self, X, model_dict=None):
        """
        Predict daily precipitation using saved two-stage model.
        Returns: combined_pred, wet_proba
        """
        if model_dict is None:
            model_dict = joblib.load(self.model_cache)
        clf = model_dict["classifier"]
        reg = model_dict["regressor"]
        thr = model_dict["decision_threshold"]
        # Align columns if needed
        feat_order = model_dict.get("feature_names", list(X.columns))
        # Add missing columns with default value 0
        for col in feat_order:
            if col not in X.columns:
                X[col] = 0

        X = X[feat_order]

        proba = clf.predict_proba(X)[:, 1]
        is_wet = proba >= thr
        preds = np.zeros(len(X), dtype=float)
        if hasattr(reg, "predict"):
            preds[is_wet] = np.clip(reg.predict(X[is_wet]), 0, None)
        return preds, proba

    @staticmethod
    def get_ontario_parks():
        return [
            {
                "id": "algonquin",
                "name": "Algonquin",
                "lat": 45.442046,
                "lon": -78.820583,
            },
            {
                "id": "killarney",
                "name": "Killarney",
                "lat": 46.012567,
                "lon": -81.401889,
            },
            {
                "id": "sandbanks",
                "name": "Sandbanks",
                "lat": 43.911220,
                "lon": -77.241690,
            },
            {"id": "bon_echo", "name": "Bon Echo", "lat": 44.897200, "lon": -77.209500},
            {"id": "killbear", "name": "Killbear", "lat": 45.358860, "lon": -80.213440},
            # {
            #     "id": "wasaga_beach",
            #     "name": "Wasaga Beach",
            #     "lat": 44.498500,
            #     "lon": -80.047283,
            # },
            # {
            #     "id": "tobermory",
            #     "name": "Tobermory",
            #     "lat": 45.253400,
            #     "lon": -81.664500,
            # },
            # {
            #     "id": "muskoka_lakes",
            #     "name": "Muskoka Lakes",
            #     "lat": 45.116600,
            #     "lon": -79.576000,
            # },
            # {
            #     "id": "blue_mountain",
            #     "name": "Blue Mountain",
            #     "lat": 44.501100,
            #     "lon": -80.316100,
            # },
        ]

    @staticmethod
    def collect_historical_for_parks(parks, days_back=365):
        out = {}
        for p in parks:
            pred = GPMPredictor(lat=p["lat"], lon=p["lon"])
            pred.data_cache = f"data/gpm_{p['id']}.pkl"  # per-park cache
            df = pred.load_or_download_data(days_back=days_back)
            df["park_id"] = p["id"]
            df["park_name"] = p["name"]
            out[p["id"]] = df
        return out

    @staticmethod
    def build_training_set(parks_data, predictor: "GPMPredictor"):
        Xs, ys = [], []
        for _, df in parks_data.items():
            X, y, _ = predictor.create_features(df.copy())
            Xs.append(X)
            ys.append(y)
        X_all = pd.concat(Xs, axis=0).reset_index(drop=True)
        y_all = pd.concat(ys, axis=0).reset_index(drop=True)
        return X_all, y_all

    @staticmethod
    def make_next_day_features(df, feature_cols=None, target_date=None):
        df = df.sort_values("date").copy()
        past = df["precipitation"]

        feats = {}
        for lag in [1, 2, 3, 7, 14, 30]:
            feats[f"precip_lag_{lag}"] = (
                float(past.iloc[-lag]) if len(past) >= lag else 0.0
            )

        for window in [7, 14, 30]:
            if len(past) >= window:
                roll = (
                    past.rolling(window=window)
                    .agg(["mean", "std", "max", "sum"])
                    .iloc[-1]
                )
                feats[f"precip_mean_{window}d"] = float(roll["mean"])
                feats[f"precip_std_{window}d"] = float(roll["std"])
                feats[f"precip_max_{window}d"] = float(roll["max"])
                feats[f"precip_sum_{window}d"] = float(roll["sum"])
            else:
                feats[f"precip_mean_{window}d"] = 0.0
                feats[f"precip_std_{window}d"] = 0.0
                feats[f"precip_max_{window}d"] = 0.0
                feats[f"precip_sum_{window}d"] = 0.0

        if target_date is None:
            last_date = pd.to_datetime(df["date"].max())
            next_date = last_date + pd.Timedelta(days=1)
        else:
            next_date = pd.to_datetime(target_date)
        month = int(next_date.month)
        doy = int(next_date.timetuple().tm_yday)
        feats["day_of_year"] = doy
        feats["month"] = month
        feats["doy_sin"] = float(np.sin(2 * np.pi * doy / 365.25))
        feats["doy_cos"] = float(np.cos(2 * np.pi * doy / 365.25))
        feats["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
        feats["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))
        weekday = int(next_date.weekday())
        feats["weekday"] = weekday
        feats["is_weekend"] = int(weekday >= 5)

        prev1 = feats.get("precip_lag_1", 0.0)
        feats["prev_wet"] = int(prev1 > 0.1)
        cnt = 0
        for v in reversed(past.values):
            if v > 0.1:
                break
            cnt += 1
        feats["days_since_rain"] = cnt

        bins = [-np.inf, 0.1, 1.0, 5.0, np.inf]
        feats["precip_intensity_prev"] = float(
            np.digitize([prev1], bins, right=False)[0] - 1
        )

        month_hist = df[df["date"].dt.month == month]["precipitation"].iloc[:-1]
        clim = float(month_hist.mean()) if len(month_hist) else 0.0
        feats["monthly_climatology"] = clim
        feats["seasonal_anomaly"] = float(prev1 - clim)

        if "lat" in df.columns and "lon" in df.columns:
            feats["lat"] = float(df["lat"].iloc[-1])
            feats["lon"] = float(df["lon"].iloc[-1])

        feature_cols = list(feats.keys())
        X_row = pd.DataFrame([feats]).reindex(columns=feature_cols, fill_value=0.0)
        return X_row

    @staticmethod
    def forecast_next_days_for_parks(parks_data, model_dict, days=14):
        """
        Make daily forecasts for each park.
        """
        feat_cols = model_dict.get("feature_names")
        wrapper = GPMPredictor()
        rows = []
        today = pd.Timestamp.today().normalize()
        for park_id, df in parks_data.items():
            hist = df.sort_values("date").copy()
            park_name = (
                hist["park_name"].iloc[0] if "park_name" in hist.columns else park_id
            )
            for d in range(days):
                target_date = today + pd.Timedelta(days=d)
                X_row = GPMPredictor.make_next_day_features(
                    hist, feat_cols, target_date=target_date
                )
                preds, proba = wrapper.predict(X_row, model_dict=model_dict)
                rows.append(
                    {
                        "park_id": park_id,
                        "park_name": park_name,
                        "date": target_date,
                        "wet_proba": float(proba[0]),
                        "amount_mm": float(preds[0]),
                    }
                )
                # Roll history forward with predicted amount
                new_row = {
                    "date": target_date,
                    "precipitation": float(preds[0]),
                    "lat": (
                        float(hist["lat"].iloc[-1]) if "lat" in hist.columns else np.nan
                    ),
                    "lon": (
                        float(hist["lon"].iloc[-1]) if "lon" in hist.columns else np.nan
                    ),
                }
                hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        return (
            pd.DataFrame(rows).sort_values(["park_id", "date"]).reset_index(drop=True)
        )

    @staticmethod
    def forecast_next_days_for_parks(parks_data, model_dict, days=14):
        """
        Recursive multi-day forecast per park (default 14 days).
        """
        feat_cols = model_dict.get("feature_names")
        wrapper = GPMPredictor()
        rows = []
        for park_id, df in parks_data.items():
            hist = df.sort_values("date").copy()
            park_name = (
                hist["park_name"].iloc[0] if "park_name" in hist.columns else park_id
            )
            for d in range(1, days + 1):
                X_row = GPMPredictor.make_next_day_features(hist, feat_cols)
                preds, proba = wrapper.predict(X_row, model_dict=model_dict)
                last_date = hist["date"].max()
                next_date = last_date + pd.Timedelta(days=1)
                rows.append(
                    {
                        "park_id": park_id,
                        "park_name": park_name,
                        "date": next_date,
                        "wet_proba": float(proba[0]),
                        "amount_mm": float(preds[0]),
                    }
                )
                # Roll history forward with predicted amount
                new_row = {
                    "date": next_date,
                    "precipitation": float(preds[0]),
                    "lat": (
                        float(hist["lat"].iloc[-1]) if "lat" in hist.columns else np.nan
                    ),
                    "lon": (
                        float(hist["lon"].iloc[-1]) if "lon" in hist.columns else np.nan
                    ),
                }
                hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        return (
            pd.DataFrame(rows).sort_values(["park_id", "date"]).reset_index(drop=True)
        )

    @staticmethod
    def rank_parks_for_day(forecast_df, day_index=1, top_k=None):
        """
        Return parks sorted by lowest rain probability for a given day in the forecast DF produced by forecast_next_days_for_parks.
        """
        if forecast_df.empty:
            return forecast_df
        start_date = forecast_df["date"].min()
        target_date = start_date + pd.Timedelta(days=day_index - 1)
        sub = forecast_df[forecast_df["date"] == target_date].copy()
        sub = sub.sort_values(["wet_proba", "amount_mm"], ascending=[True, True])
        if top_k:
            sub = sub.head(top_k)
        return sub.reset_index(drop=True)

    @staticmethod
    def forecast_for_ui(parks_data, model_dict, days=14):
        """
        Produce JSON-ready results for UI: [{park_id, park_name, daily:[{date, wet_proba, amount_mm}, ...]}]
        """
        df = GPMPredictor.forecast_next_days_for_parks(
            parks_data, model_dict, days=days
        )
        results = []
        for park_id, group in df.groupby("park_id"):
            park_name = (
                group["park_name"].iloc[0] if "park_name" in group.columns else park_id
            )
            daily = [
                {
                    "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                    "wet_proba": float(r["wet_proba"]),
                    "amount_mm": float(r["amount_mm"]),
                }
                for _, r in group.sort_values("date").iterrows()
            ]
            results.append({"park_id": park_id, "park_name": park_name, "daily": daily})
        return results
