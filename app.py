from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Solar Intelligence Studio", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "Plant 1": (
        APP_ROOT / "Plant_1_Generation_Data.csv",
        APP_ROOT / "Plant_1_Weather_Sensor_Data.csv",
    ),
    "Plant 2": (
        APP_ROOT / "Plant_2_Generation_Data.csv",
        APP_ROOT / "Plant_2_Weather_Sensor_Data.csv",
    ),
}

IMPACT_FACTORS = {
    "home_day_kwh": 29.0,
    "ev_charge_kwh": 50.0,
    "classroom_day_kwh": 12.0,
    "home_live_kw": 1.25,
    "ev_fast_kw": 50.0,
    "classroom_live_kw": 3.0,
}


def apply_theme(theme_mode: str) -> str:
    if theme_mode == "Light":
        css = """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 196, 0, 0.18), transparent 34%),
                linear-gradient(180deg, #fffaf0 0%, #fff4dc 42%, #f7fbff 100%);
            color: #172033;
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.9);
            border-right: 1px solid rgba(23, 32, 51, 0.08);
        }
        .solar-card {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(23, 32, 51, 0.08);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .solar-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            color: #c26d00;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .solar-big {
            font-size: 1.85rem;
            font-weight: 700;
            color: #162033;
        }
        .solar-copy {
            color: #455066;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        </style>
        """
        template = "plotly_white"
    else:
        css = """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 179, 0, 0.16), transparent 28%),
                linear-gradient(180deg, #09111f 0%, #0c1628 55%, #111d31 100%);
            color: #e8edf7;
        }
        [data-testid="stSidebar"] {
            background: rgba(9, 17, 31, 0.92);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        .solar-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 18px 34px rgba(0, 0, 0, 0.24);
            margin-bottom: 14px;
        }
        .solar-kicker {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            color: #ffbe5c;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .solar-big {
            font-size: 1.85rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .solar-copy {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        </style>
        """
        template = "plotly_dark"

    st.markdown(css, unsafe_allow_html=True)
    return template


def render_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="solar-card">
            <div class="solar-kicker">{title}</div>
            <div class="solar-big">{value}</div>
            <div class="solar-copy">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def current_india_time() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ZoneInfo("Asia/Kolkata"))


def parse_datetime_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.35:
        parsed = pd.to_datetime(series, format="mixed", errors="coerce", dayfirst=False)
    return parsed


@st.cache_data(show_spinner=False)
def load_data(gen_file, weather_file) -> pd.DataFrame:
    gen = pd.read_csv(gen_file)
    weather = pd.read_csv(weather_file)

    gen["DATE_TIME"] = parse_datetime_column(gen["DATE_TIME"])
    weather["DATE_TIME"] = parse_datetime_column(weather["DATE_TIME"])
    gen = gen.rename(columns={"SOURCE_KEY": "GEN_SOURCE"})
    weather = weather.rename(columns={"SOURCE_KEY": "WEATHER_SOURCE"})

    df = pd.merge(gen, weather, on=["DATE_TIME", "PLANT_ID"], how="inner")
    df = df.dropna(subset=["DATE_TIME"]).sort_values("DATE_TIME").reset_index(drop=True)

    if df.empty:
        raise ValueError("The generation and weather files could not be merged.")

    df["DATE"] = df["DATE_TIME"].dt.normalize()
    df["YEAR"] = df["DATE_TIME"].dt.year
    df["MONTH"] = df["DATE_TIME"].dt.month
    df["WEEK"] = df["DATE_TIME"].dt.isocalendar().week.astype(int)
    df["HOUR"] = df["DATE_TIME"].dt.hour
    df["MINUTE"] = df["DATE_TIME"].dt.minute
    df["DAY_OF_WEEK"] = df["DATE_TIME"].dt.dayofweek
    df["DAY_NAME"] = df["DATE_TIME"].dt.day_name()
    df["DAY_OF_YEAR"] = df["DATE_TIME"].dt.dayofyear
    df["IS_DAYLIGHT"] = (df["IRRADIATION"] > 0).astype(int)
    df["AC_DC_RATIO"] = np.where(df["DC_POWER"] > 0, df["AC_POWER"] / df["DC_POWER"], np.nan)
    df["AC_DC_RATIO"] = df["AC_DC_RATIO"].clip(lower=0, upper=1.2)
    return df


@st.cache_data(show_spinner=False)
def build_views(df: pd.DataFrame):
    power_ts = (
        df.groupby("DATE_TIME", as_index=False)
        .agg(
            total_ac_power=("AC_POWER", "sum"),
            total_dc_power=("DC_POWER", "sum"),
            avg_irradiation=("IRRADIATION", "mean"),
            avg_ambient_temp=("AMBIENT_TEMPERATURE", "mean"),
            avg_module_temp=("MODULE_TEMPERATURE", "mean"),
            active_units=("GEN_SOURCE", "nunique"),
        )
        .sort_values("DATE_TIME")
    )
    power_ts["DATE"] = power_ts["DATE_TIME"].dt.normalize()
    power_ts["HOUR"] = power_ts["DATE_TIME"].dt.hour
    power_ts["MINUTE"] = power_ts["DATE_TIME"].dt.minute
    power_ts["DAY_OF_WEEK"] = power_ts["DATE_TIME"].dt.dayofweek
    power_ts["MONTH"] = power_ts["DATE_TIME"].dt.month
    power_ts["DAY_OF_YEAR"] = power_ts["DATE_TIME"].dt.dayofyear
    power_ts["PLANT_EFFICIENCY"] = np.where(
        power_ts["total_dc_power"] > 0,
        power_ts["total_ac_power"] / power_ts["total_dc_power"],
        np.nan,
    )
    power_ts["PLANT_EFFICIENCY"] = power_ts["PLANT_EFFICIENCY"].clip(lower=0, upper=1.2)

    inverter_daily = (
        df.groupby(["DATE", "GEN_SOURCE"], as_index=False)["DAILY_YIELD"].max()
        .rename(columns={"DAILY_YIELD": "inverter_daily_yield"})
    )
    daily_energy = (
        inverter_daily.groupby("DATE", as_index=False)["inverter_daily_yield"]
        .sum()
        .rename(columns={"inverter_daily_yield": "daily_energy_kwh"})
    )

    daily_weather = (
        power_ts.groupby("DATE", as_index=False)
        .agg(
            avg_irradiation=("avg_irradiation", "mean"),
            avg_module_temp=("avg_module_temp", "mean"),
            avg_ambient_temp=("avg_ambient_temp", "mean"),
            avg_ac_power=("total_ac_power", "mean"),
            peak_ac_power=("total_ac_power", "max"),
            avg_efficiency=("PLANT_EFFICIENCY", "mean"),
        )
    )
    daily = daily_energy.merge(daily_weather, on="DATE", how="left")
    daily["rolling_7d_energy"] = daily["daily_energy_kwh"].rolling(7, min_periods=1).mean()
    daily["DAY_NAME"] = daily["DATE"].dt.day_name()
    daily["DAY_OF_WEEK"] = daily["DATE"].dt.dayofweek
    daily["MONTH"] = daily["DATE"].dt.month

    hourly = (
        power_ts.groupby(["DATE", "HOUR"], as_index=False)
        .agg(
            avg_ac_power=("total_ac_power", "mean"),
            avg_dc_power=("total_dc_power", "mean"),
            avg_irradiation=("avg_irradiation", "mean"),
            avg_module_temp=("avg_module_temp", "mean"),
            avg_ambient_temp=("avg_ambient_temp", "mean"),
        )
        .sort_values(["DATE", "HOUR"])
    )
    hourly["plant_efficiency"] = np.where(
        hourly["avg_dc_power"] > 0,
        hourly["avg_ac_power"] / hourly["avg_dc_power"],
        np.nan,
    )
    hourly["plant_efficiency"] = hourly["plant_efficiency"].clip(lower=0, upper=1.2)

    return power_ts, daily, hourly


def apply_display_timeline(power_ts: pd.DataFrame, daily: pd.DataFrame, hourly: pd.DataFrame, current_time: pd.Timestamp):
    current_naive = current_time.tz_localize(None) if current_time.tzinfo is not None else current_time
    latest_internal_ts = power_ts["DATE_TIME"].max()
    display_shift = current_naive.normalize() - latest_internal_ts.normalize()

    power_view = power_ts.copy()
    daily_view = daily.copy()
    hourly_view = hourly.copy()

    power_view["DISPLAY_DATE_TIME"] = power_view["DATE_TIME"] + display_shift
    power_view["DISPLAY_DATE"] = power_view["DISPLAY_DATE_TIME"].dt.normalize()

    daily_view["DISPLAY_DATE"] = daily_view["DATE"] + display_shift
    daily_view["DISPLAY_YEAR"] = daily_view["DISPLAY_DATE"].dt.year

    hourly_view["DISPLAY_DATE"] = hourly_view["DATE"] + display_shift
    hourly_view["DISPLAY_DATE_TIME"] = hourly_view["DISPLAY_DATE"] + pd.to_timedelta(hourly_view["HOUR"], unit="h")
    hourly_view["window_label"] = hourly_view["DISPLAY_DATE_TIME"].dt.strftime("%Y-%m-%d %H:00")

    return power_view, daily_view, hourly_view


@st.cache_resource(show_spinner=False)
def train_power_model(power_ts: pd.DataFrame):
    training = power_ts[power_ts["avg_irradiation"] > 0].copy()
    if len(training) < 50:
        raise ValueError("Not enough daylight data points to train the prediction model.")

    feature_columns = [
        "avg_irradiation",
        "avg_ambient_temp",
        "avg_module_temp",
        "HOUR",
        "MINUTE",
        "DAY_OF_WEEK",
        "MONTH",
        "DAY_OF_YEAR",
    ]
    X = training[feature_columns]
    y = training["total_ac_power"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=220,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    residual_std = float(np.std(y_test - predictions))

    return {
        "model": model,
        "features": feature_columns,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "residual_std": residual_std,
        "sample_size": int(len(training)),
    }


@st.cache_data(show_spinner=False)
def score_anomalies(hourly: pd.DataFrame) -> pd.DataFrame:
    working = hourly[hourly["avg_irradiation"] > 0].copy()
    if len(working) < 25:
        working["anomaly_flag"] = 1
        working["risk_score"] = 0.0
        return working

    feature_columns = [
        "avg_ac_power",
        "avg_irradiation",
        "avg_module_temp",
        "avg_ambient_temp",
        "plant_efficiency",
    ]
    detector = IsolationForest(
        n_estimators=120,
        contamination=0.08,
        random_state=42,
    )
    detector.fit(working[feature_columns].fillna(0))
    working["anomaly_flag"] = detector.predict(working[feature_columns].fillna(0))
    raw_risk = -detector.decision_function(working[feature_columns].fillna(0))
    working["risk_score"] = np.clip(raw_risk, a_min=0, a_max=None)
    working["bubble_size"] = 10 + (working["risk_score"] * 120)
    label_source = "DISPLAY_DATE_TIME" if "DISPLAY_DATE_TIME" in working.columns else "DATE"
    working["window_label"] = working[label_source].dt.strftime("%Y-%m-%d %H:00")
    return working.sort_values("risk_score", ascending=False)


def predict_power(model_bundle, irradiation, ambient_temp, module_temp, timestamp):
    row = pd.DataFrame(
        [
            {
                "avg_irradiation": irradiation,
                "avg_ambient_temp": ambient_temp,
                "avg_module_temp": module_temp,
                "HOUR": timestamp.hour,
                "MINUTE": timestamp.minute,
                "DAY_OF_WEEK": timestamp.dayofweek,
                "MONTH": timestamp.month,
                "DAY_OF_YEAR": timestamp.dayofyear,
            }
        ]
    )
    prediction = float(model_bundle["model"].predict(row[model_bundle["features"]])[0])
    return max(prediction, 0.0)


def future_energy_projection(
    daily: pd.DataFrame,
    years_ahead: int,
    annual_change_pct: float,
    weather_shift_pct: float,
    optimization_pct: float,
) -> pd.DataFrame:
    baseline = daily.copy()
    baseline["weekday"] = baseline["DATE"].dt.dayofweek
    baseline["month"] = baseline["DATE"].dt.month

    base_mean = float(baseline["daily_energy_kwh"].mean())
    weekday_factor = (
        baseline.groupby("weekday")["daily_energy_kwh"].mean() / base_mean
    ).to_dict()
    month_factor = (
        baseline.groupby("month")["daily_energy_kwh"].mean() / base_mean
    ).to_dict()

    current_year = pd.Timestamp.today().year
    start_date = pd.Timestamp(year=current_year, month=1, day=1)
    end_date = pd.Timestamp(year=current_year + years_ahead - 1, month=12, day=31)
    future_dates = pd.date_range(start_date, end_date, freq="D")

    projected = []
    weather_multiplier = 1 + weather_shift_pct / 100.0
    optimization_multiplier = 1 + optimization_pct / 100.0

    for date in future_dates:
        year_index = date.year - current_year
        annual_multiplier = (1 + annual_change_pct / 100.0) ** year_index
        expected = (
            base_mean
            * weekday_factor.get(date.dayofweek, 1.0)
            * month_factor.get(date.month, 1.0)
            * annual_multiplier
            * weather_multiplier
            * optimization_multiplier
        )
        projected.append({"DATE": date, "projected_energy_kwh": max(expected, 0.0)})

    projection = pd.DataFrame(projected)
    projection["YEAR"] = projection["DATE"].dt.year
    return projection


def energy_equivalents(energy_kwh: float) -> dict:
    return {
        "home_days": energy_kwh / IMPACT_FACTORS["home_day_kwh"],
        "ev_charges": energy_kwh / IMPACT_FACTORS["ev_charge_kwh"],
        "classroom_days": energy_kwh / IMPACT_FACTORS["classroom_day_kwh"],
    }


def power_equivalents(power_kw: float) -> dict:
    return {
        "homes_live": power_kw / IMPACT_FACTORS["home_live_kw"],
        "ev_fast_live": power_kw / IMPACT_FACTORS["ev_fast_kw"],
        "classrooms_live": power_kw / IMPACT_FACTORS["classroom_live_kw"],
    }


def build_insights(daily: pd.DataFrame, anomalies: pd.DataFrame) -> list[str]:
    peak_day = daily.loc[daily["daily_energy_kwh"].idxmax()]
    low_day = daily.loc[daily["daily_energy_kwh"].idxmin()]
    irradiation_corr = daily["daily_energy_kwh"].corr(daily["avg_irradiation"])
    heat_corr = daily["daily_energy_kwh"].corr(daily["avg_module_temp"])
    anomaly_share = (
        float((anomalies["anomaly_flag"] == -1).mean()) if not anomalies.empty else 0.0
    )

    insights = [
        (
            f"Top production day in the current operating view reached {peak_day['daily_energy_kwh']:,.0f} kWh."
        ),
        (
            f"Lowest production day in the current operating view was {low_day['daily_energy_kwh']:,.0f} kWh, which is useful as a stress-case baseline."
        ),
        (
            f"Daily energy and irradiation move together with a correlation of {irradiation_corr:.2f}, "
            "so weather-aware prediction is materially better than a static formula."
        ),
    ]

    if pd.notna(heat_corr) and heat_corr < 0:
        insights.append(
            f"Module temperature shows a negative correlation of {heat_corr:.2f} with daily energy, "
            "which supports heat-loss monitoring and cooling or cleaning interventions."
        )
    else:
        insights.append(
            "The short history does not show a strong negative temperature signature yet, "
            "so maintenance recommendations should be paired with live monitoring."
        )

    insights.append(
        f"About {anomaly_share * 100:.1f}% of daylight hourly windows are flagged as unusual, "
        "giving the app a simple offline AI layer for early operational alerts."
    )
    return insights


def build_operational_suggestions(hourly: pd.DataFrame, flagged: pd.DataFrame) -> list[str]:
    if flagged.empty:
        return [
            "No high-risk daylight anomalies were detected in the current sample, so the plant looks operationally stable.",
            "Keep monitoring conversion efficiency and compare future uploads against this baseline.",
        ]

    suggestions = []
    hot_threshold = hourly["avg_module_temp"].quantile(0.75)
    flagged_hot_share = (flagged["avg_module_temp"] >= hot_threshold).mean()
    if flagged_hot_share >= 0.4:
        suggestions.append(
            "Many flagged windows happen during high module-temperature periods, so panel cleaning, airflow, and hotspot inspection are the first checks to run."
        )

    flagged_efficiency = flagged["avg_ac_power"] / flagged["avg_irradiation"].replace(0, np.nan)
    normal = hourly[hourly["avg_irradiation"] > 0].copy()
    normal_efficiency = normal["avg_ac_power"] / normal["avg_irradiation"].replace(0, np.nan)
    if flagged_efficiency.median(skipna=True) < normal_efficiency.median(skipna=True) * 0.8:
        suggestions.append(
            "Flagged windows are converting sunlight into AC power less efficiently than normal periods, which points to inverter, string, or soiling losses."
        )

    clustered_hours = flagged["HOUR"].mode()
    if not clustered_hours.empty:
        suggestions.append(
            f"Most anomalies cluster around {int(clustered_hours.iloc[0]):02d}:00, so that hour is the best place to inspect recurring operational issues."
        )

    if len(suggestions) < 2:
        suggestions.append(
            "Use the top-risk windows table as a maintenance shortlist, because those timestamps combine unusual behavior with measurable weather context."
        )
    return suggestions[:3]


def resolve_display_date(selected_date, available_dates) -> pd.Timestamp:
    target = pd.Timestamp(selected_date).normalize()
    ordered_dates = pd.Series(pd.to_datetime(sorted(available_dates))).sort_values().reset_index(drop=True)

    if target in set(ordered_dates):
        return target

    same_weekday = ordered_dates[ordered_dates.dt.dayofweek == target.dayofweek]
    candidate_pool = same_weekday if not same_weekday.empty else ordered_dates
    day_distance = (candidate_pool.dt.dayofyear - target.dayofyear).abs()
    best_idx = int(day_distance.idxmin())
    return pd.Timestamp(candidate_pool.loc[best_idx]).normalize()


st.sidebar.title("Solar Intelligence")
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

selected_theme = st.sidebar.selectbox(
    "Theme",
    ["Dark", "Light"],
    index=0 if st.session_state.theme_mode == "Dark" else 1,
)
st.session_state.theme_mode = selected_theme
PLOTLY_TEMPLATE = apply_theme(selected_theme)

default_plant = st.sidebar.selectbox("Default plant dataset", list(DEFAULT_FILES.keys()), index=1)
default_gen, default_weather = DEFAULT_FILES[default_plant]
uploaded_gen = st.sidebar.file_uploader("Upload generation CSV", type=["csv"])
uploaded_weather = st.sidebar.file_uploader("Upload weather CSV", type=["csv"])

gen_source = uploaded_gen if uploaded_gen is not None else default_gen
weather_source = uploaded_weather if uploaded_weather is not None else default_weather

raw_df = power_ts = daily = hourly = model_bundle = anomalies = None
try:
    raw_df = load_data(gen_source, weather_source)
    power_ts, daily, hourly = build_views(raw_df)
    india_now = current_india_time()
    power_ts, daily, hourly = apply_display_timeline(power_ts, daily, hourly, india_now)
    model_bundle = train_power_model(power_ts)
    anomalies = score_anomalies(hourly)
except Exception as exc:
    st.error(f"Unable to prepare the AI dashboard: {exc}")
    st.stop()

if any(item is None for item in [raw_df, power_ts, daily, hourly, model_bundle, anomalies]):
    raise RuntimeError("Dashboard initialization did not complete successfully.")

latest_date = daily["DATE"].max()
insights = build_insights(daily, anomalies)

st.title("Solar Intelligence Studio")
st.caption(
    "Notebook stays untouched. This Streamlit app now uses offline predictive analytics, "
    "operational anomaly detection, and multi-year planning forecasts."
)

hero_left, hero_right = st.columns([1.3, 1])
with hero_left:
    render_card(
        "Live India time",
        india_now.strftime("%d %b %Y, %I:%M %p"),
        "Current real timestamp for the app session.",
    )
with hero_right:
    render_card(
        "Loaded plant",
        default_plant if uploaded_gen is None else "Custom upload",
        f"{len(power_ts):,} plant-level timestamps and {len(daily):,} daily energy records.",
    )

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "AI Prediction", "Operations AI", "Future Planning"],
)

if page == "Overview":
    available_dates = sorted(daily["DISPLAY_DATE"].dt.date.unique())
    selected_date = st.date_input(
        "Select a plant date",
        value=india_now.date(),
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )
    selected_ts = resolve_display_date(selected_date, available_dates)
    filtered_power = power_ts[power_ts["DISPLAY_DATE"] == selected_ts]
    filtered_daily = daily[daily["DISPLAY_DATE"] == selected_ts]
    week_context = daily[
        (daily["DISPLAY_DATE"] >= selected_ts - pd.Timedelta(days=3))
        & (daily["DISPLAY_DATE"] <= selected_ts + pd.Timedelta(days=3))
    ].copy()

    if filtered_power.empty or filtered_daily.empty:
        st.warning("No records are available for the selected date.")
        st.stop()

    total_energy = float(filtered_daily["daily_energy_kwh"].sum())
    avg_power = float(filtered_power["total_ac_power"].mean())
    peak_power = float(filtered_power["total_ac_power"].max())
    avg_efficiency = float(filtered_power["PLANT_EFFICIENCY"].dropna().mean())

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        render_card("Energy delivered", f"{total_energy:,.0f} kWh", f"Selected date: {selected_date}")
    with card2:
        render_card("Average AC power", f"{avg_power:,.1f} kW", "Plant-level mean output.")
    with card3:
        render_card("Peak AC power", f"{peak_power:,.1f} kW", "Highest observed power in range.")
    with card4:
        render_card("AC/DC ratio", f"{avg_efficiency:.2f}", "Useful proxy for conversion health.")

    charts_col, notes_col = st.columns([1.7, 1])
    with charts_col:
        fig = px.line(
            filtered_power,
            x="DISPLAY_DATE_TIME",
            y=["total_ac_power", "avg_irradiation"],
            template=PLOTLY_TEMPLATE,
            title="Plant output and irradiation",
        )
        st.plotly_chart(fig, use_container_width=True)

        energy_fig = px.bar(
            week_context,
            x="DISPLAY_DATE",
            y="daily_energy_kwh",
            color="avg_module_temp",
            color_continuous_scale="YlOrBr",
            template=PLOTLY_TEMPLATE,
            title="Selected-day context across nearby dates",
        )
        st.plotly_chart(energy_fig, use_container_width=True)
    with notes_col:
        for idx, insight in enumerate(insights, start=1):
            render_card(f"AI insight {idx}", "", insight)

elif page == "AI Prediction":
    st.subheader("Offline power prediction")
    left, right = st.columns([1.2, 1])

    with left:
        st.write(
            "This replaces the old mock formula with a trained model built from your plant's "
            "weather and power history."
        )
        prediction_time = st.slider("Hour of day", 0, 23, 13)
        irradiation = st.slider("Irradiation", 0.0, 1.2, 0.65, 0.01)
        ambient_temp = st.slider("Ambient temperature (C)", 15.0, 45.0, 30.0, 0.5)
        module_temp = st.slider("Module temperature (C)", 15.0, 75.0, 42.0, 0.5)
        prediction_date = st.date_input("Forecast date context", india_now.date())

        timestamp = pd.Timestamp(prediction_date) + pd.Timedelta(hours=prediction_time)
        predicted_power = predict_power(
            model_bundle,
            irradiation=irradiation,
            ambient_temp=ambient_temp,
            module_temp=module_temp,
            timestamp=timestamp,
        )
        optimized_power = predict_power(
            model_bundle,
            irradiation=irradiation * 1.05,
            ambient_temp=max(ambient_temp - 1.5, 0),
            module_temp=max(module_temp - 4, 0),
            timestamp=timestamp,
        )

        band = model_bundle["residual_std"]
        st.success(
            f"Predicted plant AC power: {predicted_power:,.1f} kW "
            f"(typical uncertainty band +/- {band:,.1f} kW)"
        )
        st.info(
            f"If conditions improve slightly, the same model estimates about {optimized_power:,.1f} kW."
        )

        equivalents = power_equivalents(predicted_power)
        eq1, eq2, eq3 = st.columns(3)
        eq1.metric("Homes supported now", f"{equivalents['homes_live']:.1f}")
        eq2.metric("Fast chargers now", f"{equivalents['ev_fast_live']:.1f}")
        eq3.metric("Classrooms now", f"{equivalents['classrooms_live']:.1f}")
        st.caption(
            "Instant equivalents use power-based assumptions: about 1.25 kW per home, 50 kW per fast charger, and 3 kW per classroom."
        )

    with right:
        render_card(
            "Model sample size",
            f"{model_bundle['sample_size']:,}",
            "Daylight plant-level records used for training.",
        )
        render_card(
            "Mean absolute error",
            f"{model_bundle['mae']:.1f} kW",
            "Lower is better.",
        )
        render_card(
            "R-squared",
            f"{model_bundle['r2']:.3f}",
            "Explained variance on holdout data.",
        )

        comparison = pd.DataFrame(
            {
                "scenario": ["Current conditions", "Improved cooling and irradiation"],
                "predicted_power": [predicted_power, optimized_power],
            }
        )
        comparison_fig = px.bar(
            comparison,
            x="scenario",
            y="predicted_power",
            color="scenario",
            template=PLOTLY_TEMPLATE,
            title="Scenario comparison",
        )
        st.plotly_chart(comparison_fig, use_container_width=True)

elif page == "Operations AI":
    st.subheader("Operational intelligence")
    flagged = anomalies[anomalies["anomaly_flag"] == -1].copy()
    ops_suggestions = build_operational_suggestions(hourly, flagged)

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Flagged windows", f"{len(flagged)}")
    metric_b.metric(
        "Highest risk score",
        f"{flagged['risk_score'].max():.3f}" if not flagged.empty else "0.000",
    )
    metric_c.metric(
        "Average daylight efficiency",
        f"{hourly.loc[hourly['avg_irradiation'] > 0, 'plant_efficiency'].mean():.2f}",
    )

    scatter = px.scatter(
        anomalies,
        x="avg_irradiation",
        y="avg_ac_power",
        color=anomalies["anomaly_flag"].map({1: "normal", -1: "flagged"}),
        size="bubble_size",
        hover_name="window_label",
        template=PLOTLY_TEMPLATE,
        title="Hourly operational windows",
        labels={"color": "status"},
    )
    st.plotly_chart(scatter, use_container_width=True)

    table_col, actions_col = st.columns([1.4, 1])
    with table_col:
        st.write("Top unusual windows")
        display_table = flagged[
            ["window_label", "avg_ac_power", "avg_irradiation", "avg_module_temp", "risk_score"]
        ].head(10)
        st.dataframe(display_table, use_container_width=True, hide_index=True)
    with actions_col:
        hottest_flag = flagged["avg_module_temp"].max() if not flagged.empty else hourly["avg_module_temp"].max()
        low_eff_hours = hourly.nsmallest(3, "plant_efficiency")[["DISPLAY_DATE_TIME", "plant_efficiency"]]

        render_card(
            "Maintenance hint",
            "Thermal watch",
            ops_suggestions[0],
        )
        render_card(
            "Low efficiency windows",
            f"{len(low_eff_hours)} notable slots",
            ops_suggestions[1] if len(ops_suggestions) > 1 else "Focus inverter inspection on the lowest conversion periods visible in the table.",
        )
        if len(ops_suggestions) > 2:
            render_card("AI recommendation", "Priority check", ops_suggestions[2])

elif page == "Future Planning":
    st.subheader("Multi-year planning forecast")
    st.caption(
        "This is a scenario-based forecast. The source data covers about one month, so future years "
        "should be treated as planning guidance rather than a bank-grade financial forecast."
    )

    controls_left, controls_right = st.columns(2)
    with controls_left:
        years_ahead = st.slider("Years to project", 1, 10, 5)
        annual_change_pct = st.slider("Annual performance change (%)", -8.0, 8.0, -1.0, 0.5)
    with controls_right:
        weather_shift_pct = st.slider("Weather outlook adjustment (%)", -15.0, 15.0, 0.0, 0.5)
        optimization_pct = st.slider("Operational optimization uplift (%)", 0.0, 20.0, 4.0, 0.5)

    projection = future_energy_projection(
        daily,
        years_ahead=years_ahead,
        annual_change_pct=annual_change_pct,
        weather_shift_pct=weather_shift_pct,
        optimization_pct=optimization_pct,
    )
    annual_projection = (
        projection.groupby("YEAR", as_index=False)["projected_energy_kwh"]
        .sum()
        .rename(columns={"projected_energy_kwh": "annual_energy_kwh"})
    )
    annual_projection["rolling_change_pct"] = annual_projection["annual_energy_kwh"].pct_change().mul(100)

    latest_projected_total = float(annual_projection.iloc[-1]["annual_energy_kwh"])
    equivalents = energy_equivalents(latest_projected_total)

    top1, top2, top3 = st.columns(3)
    top1.metric("Final projected year", f"{latest_projected_total:,.0f} kWh")
    top2.metric("Home-days in final year", f"{equivalents['home_days']:,.0f}")
    top3.metric("EV charges in final year", f"{equivalents['ev_charges']:,.0f}")

    annual_fig = px.line(
        annual_projection,
        x="YEAR",
        y="annual_energy_kwh",
        markers=True,
        template=PLOTLY_TEMPLATE,
        title="Projected annual energy",
    )
    st.plotly_chart(annual_fig, use_container_width=True)

    daily_fig = px.line(
        projection.head(365),
        x="DATE",
        y="projected_energy_kwh",
        template=PLOTLY_TEMPLATE,
        title="First projected year daily profile",
    )
    st.plotly_chart(daily_fig, use_container_width=True)

    st.dataframe(
        annual_projection.assign(
            annual_energy_kwh=lambda frame: frame["annual_energy_kwh"].round(0),
            rolling_change_pct=lambda frame: frame["rolling_change_pct"].round(2),
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Energy equivalents use 29 kWh per home-day, 50 kWh per EV charge, and 12 kWh per classroom-day."
    )
