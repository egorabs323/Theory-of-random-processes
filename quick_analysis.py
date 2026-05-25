from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "wordstat_dynamic.xlsx"
SEASONAL_PERIOD = 12
ARIMA_P_RANGE = range(0, 4)
ARIMA_D_RANGE = range(0, 3)
ARIMA_Q_RANGE = range(0, 4)
SARIMA_P_RANGE = range(0, 3)
SARIMA_Q_RANGE = range(0, 3)
SARIMA_SP_RANGE = range(0, 2)
SARIMA_SD_RANGE = range(0, 2)
SARIMA_SQ_RANGE = range(0, 2)


@dataclass
class ModelResult:
    name: str
    family: str
    rationale: str
    params_text: str
    fitted_params: dict
    forecast: pd.Series
    residuals: pd.Series
    metrics: dict[str, float]
    aic: float | None
    bic: float | None
    ljung_box_pvalue: float | None
    diagnostic_figure: str = ""


def print_section(number: int, title: str) -> None:
    print(f"\n({number}) {title}")
    print("-" * 90)


def ensure_figures_dir() -> Path:
    figures_dir = BASE_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def choose_date_column(df: pd.DataFrame) -> str:
    datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if datetime_cols:
        return datetime_cols[0]

    for col in df.columns:
        if "период" in str(col).lower():
            return col
    raise ValueError("Не удалось определить столбец с датой.")


def choose_value_column(df: pd.DataFrame, date_col: str) -> str:
    preferred = [col for col in df.columns if "запрос" in str(col).lower()]
    numeric_cols = [col for col in df.columns if col != date_col and pd.api.types.is_numeric_dtype(df[col])]
    for col in preferred:
        if col in numeric_cols:
            return col
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError("Не удалось определить числовой столбец временного ряда.")


def load_series(path: Path) -> tuple[pd.Series, str, str, int]:
    df = pd.read_excel(path)
    date_col = choose_date_column(df)
    value_col = choose_value_column(df, date_col)

    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna().sort_values(date_col).drop_duplicates(subset=date_col)

    ts = work.set_index(date_col)[value_col].astype(float).asfreq("MS")
    missing_before_fill = int(ts.isna().sum())
    if missing_before_fill:
        ts = ts.interpolate(method="time").ffill().bfill()

    ts.name = value_col
    return ts, date_col, value_col, missing_before_fill


def train_test_split_series(ts: pd.Series, test_share: float = 0.2) -> tuple[pd.Series, pd.Series]:
    split_idx = int(len(ts) * (1 - test_share))
    return ts.iloc[:split_idx], ts.iloc[split_idx:]


def safe_mape(actual: pd.Series, predicted: pd.Series) -> float:
    actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
    mask = actual_aligned.abs() > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual_aligned[mask] - predicted_aligned[mask]) / actual_aligned[mask])) * 100)


def compute_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_aligned, predicted_aligned = actual.align(predicted, join="inner")
    mae = float(mean_absolute_error(actual_aligned, predicted_aligned))
    mse = float(mean_squared_error(actual_aligned, predicted_aligned))
    rmse = float(np.sqrt(mse))
    mape = safe_mape(actual_aligned, predicted_aligned)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}


def detect_outliers_mad(ts: pd.Series, threshold: float = 3.5) -> tuple[pd.Series, pd.Series, float, float]:
    median = float(ts.median())
    mad = float(np.median(np.abs(ts - median)))
    if mad == 0:
        modified_z = pd.Series(0.0, index=ts.index)
    else:
        modified_z = 0.6745 * (ts - median) / mad
    mask = modified_z.abs() > threshold
    return mask, modified_z, median, mad


def stationarity_summary(ts: pd.Series) -> dict:
    diff1 = ts.diff().dropna()
    adf_orig = adfuller(ts)
    kpss_orig = kpss(ts, regression="c", nlags="auto")
    adf_diff = adfuller(diff1)
    kpss_diff = kpss(diff1, regression="c", nlags="auto")

    if adf_orig[1] < 0.05 and kpss_orig[1] > 0.05:
        suggested_d = 0
        conclusion = "Исходный ряд можно считать стационарным."
    elif adf_diff[1] < 0.05 and kpss_diff[1] > 0.05:
        suggested_d = 1
        conclusion = "Первая разность делает ряд стационарным, поэтому для ARIMA выбираем d = 1."
    else:
        suggested_d = 1
        conclusion = "Тесты противоречивы, но первая разность заметно улучшает стационарность, поэтому берется d = 1."

    return {
        "original": {"ADF_stat": adf_orig[0], "ADF_p": adf_orig[1], "KPSS_stat": kpss_orig[0], "KPSS_p": kpss_orig[1]},
        "diff1": {"ADF_stat": adf_diff[0], "ADF_p": adf_diff[1], "KPSS_stat": kpss_diff[0], "KPSS_p": kpss_diff[1]},
        "suggested_d": suggested_d,
        "conclusion": conclusion,
    }


def significant_lags(values: np.ndarray, threshold: float, limit: int = 8) -> list[int]:
    lags: list[int] = []
    for lag, value in enumerate(values[1:], start=1):
        if abs(value) > threshold:
            lags.append(lag)
        if len(lags) >= limit:
            break
    return lags


def save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    final_path = path
    try:
        fig.savefig(final_path, dpi=300, bbox_inches="tight")
    except PermissionError:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        fig.savefig(final_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return final_path


def plot_visual_analysis(ts: pd.Series, train: pd.Series, test: pd.Series, figures_dir: Path) -> str:
    fig, axes = plt.subplots(3, 1, figsize=(15, 14))
    rolling_mean = ts.rolling(window=12, min_periods=3).mean()

    axes[0].plot(ts.index, ts.values, linewidth=2, color="#1f4e79", label="Исходный ряд")
    axes[0].plot(rolling_mean.index, rolling_mean.values, linewidth=2, color="#d77a1f", label="12-месячное скользящее среднее")
    axes[0].set_title("Динамика запросов и сглаженный тренд")
    axes[0].set_ylabel("Число запросов")
    axes[0].legend()

    axes[1].plot(train.index, train.values, linewidth=2, color="#2e8b57", label="Обучающая выборка")
    axes[1].plot(test.index, test.values, linewidth=2, color="#c0392b", label="Тестовая выборка")
    axes[1].axvline(test.index[0], linestyle="--", color="black", alpha=0.6)
    axes[1].set_title("Разделение на обучающую и тестовую части")
    axes[1].set_ylabel("Число запросов")
    axes[1].legend()

    month_df = pd.DataFrame({"value": ts.values, "month": ts.index.month_name(locale="ru_RU") if False else ts.index.month})
    month_df["month_name"] = ts.index.strftime("%b")
    sns.boxplot(data=month_df, x="month_name", y="value", ax=axes[2], color="#8fb9d4")
    axes[2].set_title("Распределение значений по месяцам года")
    axes[2].set_xlabel("Месяц")
    axes[2].set_ylabel("Число запросов")

    for ax in axes:
        ax.grid(True, alpha=0.25)

    path = figures_dir / "01_visual_analysis.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def plot_outliers(ts: pd.Series, outlier_mask: pd.Series, modified_z: pd.Series, figures_dir: Path) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    axes[0].plot(ts.index, ts.values, linewidth=2, color="#1f4e79", label="Ряд")
    axes[0].scatter(
        ts.index[outlier_mask],
        ts[outlier_mask],
        color="#c0392b",
        s=80,
        zorder=5,
        label=f"Выбросы MAD ({int(outlier_mask.sum())})",
    )
    axes[0].set_title("Выбросы по критерию модифицированного z-score (MAD)")
    axes[0].set_ylabel("Число запросов")
    axes[0].legend()

    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].axhline(3.5, color="#c0392b", linestyle="--", alpha=0.7, label="Порог +3.5")
    axes[1].axhline(-3.5, color="#c0392b", linestyle="--", alpha=0.7, label="Порог -3.5")
    axes[1].plot(modified_z.index, modified_z.values, linewidth=1.8, color="#7d3c98")
    axes[1].set_title("Модифицированные z-оценки")
    axes[1].set_ylabel("Modified z-score")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.25)

    path = figures_dir / "02_outlier_analysis.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def plot_acf_pacf_overview(ts: pd.Series, figures_dir: Path) -> tuple[str, dict]:
    diff1 = ts.diff().dropna()
    max_lags = min(24, max(8, len(ts) // 3))
    pacf_lags = min(max_lags, len(diff1) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_acf(ts, lags=max_lags, ax=axes[0, 0])
    axes[0, 0].set_title("ACF исходного ряда")
    plot_pacf(ts, lags=min(max_lags, len(ts) // 2 - 1), ax=axes[0, 1], method="ywm")
    axes[0, 1].set_title("PACF исходного ряда")
    plot_acf(diff1, lags=max_lags, ax=axes[1, 0])
    axes[1, 0].set_title("ACF первой разности")
    plot_pacf(diff1, lags=pacf_lags, ax=axes[1, 1], method="ywm")
    axes[1, 1].set_title("PACF первой разности")

    path = figures_dir / "03_acf_pacf.png"
    saved_path = save_figure(fig, path)

    acf_vals = pd.Series(statsmodels_acf(diff1, nlags=max_lags), index=range(max_lags + 1))
    pacf_vals = pd.Series(statsmodels_pacf(diff1, nlags=pacf_lags), index=range(pacf_lags + 1))
    threshold = 1.96 / np.sqrt(len(diff1))
    analysis = {
        "threshold": threshold,
        "diff_acf_lags": significant_lags(acf_vals.values, threshold),
        "diff_pacf_lags": significant_lags(pacf_vals.values, threshold),
    }
    return saved_path.name, analysis


def statsmodels_acf(series: pd.Series, nlags: int) -> np.ndarray:
    from statsmodels.tsa.stattools import acf

    return acf(series, nlags=nlags, fft=False)


def statsmodels_pacf(series: pd.Series, nlags: int) -> np.ndarray:
    from statsmodels.tsa.stattools import pacf

    return pacf(series, nlags=nlags, method="ywm")


def run_arima_grid_search(train: pd.Series, test: pd.Series, suggested_d: int) -> tuple[pd.DataFrame, ModelResult]:
    rows: list[dict] = []
    for p in ARIMA_P_RANGE:
        for d in ARIMA_D_RANGE:
            for q in ARIMA_Q_RANGE:
                try:
                    fitted = ARIMA(
                        train,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit()
                    forecast = fitted.get_forecast(steps=len(test)).predicted_mean
                    forecast.index = test.index
                    metrics = compute_metrics(test, forecast)
                    rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "MAE": metrics["MAE"],
                            "MSE": metrics["MSE"],
                            "RMSE": metrics["RMSE"],
                            "MAPE": metrics["MAPE"],
                            "AIC": float(fitted.aic),
                            "BIC": float(fitted.bic),
                        }
                    )
                except Exception:
                    continue

    if not rows:
        raise RuntimeError("ARIMA grid search не дал ни одной корректной модели.")

    grid_df = pd.DataFrame(rows).sort_values(["MAE", "AIC", "BIC"]).reset_index(drop=True)
    preferred = grid_df[grid_df["d"] == suggested_d]
    best_row = preferred.iloc[0] if not preferred.empty else grid_df.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))

    fitted = ARIMA(
        train,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit()
    forecast = fitted.get_forecast(steps=len(test)).predicted_mean
    forecast.index = test.index
    metrics = compute_metrics(test, forecast)
    residuals = pd.Series(fitted.resid, index=train.index)
    lb = acorr_ljungbox(residuals, lags=[min(10, len(residuals) // 3)], return_df=True)["lb_pvalue"].iloc[0]
    fitted_params = {name: round(float(value), 6) for name, value in fitted.params.items()}

    model = ModelResult(
        name="ARIMA",
        family="ARIMA",
        rationale=(
            "По лекции ACF помогает выбирать q, а PACF - p. После первой разности значимы только низкие лаги, "
            "поэтому тестируется семейство низких порядков p, q <= 3. Параметр d выбирается тестами стационарности."
        ),
        params_text=f"order = {best_order}",
        fitted_params={"order": best_order, "coefficients": fitted_params},
        forecast=forecast,
        residuals=residuals,
        metrics=metrics,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        ljung_box_pvalue=float(lb),
    )
    return grid_df, model


def run_sarima_grid_search(train: pd.Series, test: pd.Series, suggested_d: int) -> tuple[pd.DataFrame, ModelResult]:
    rows: list[dict] = []
    for p in SARIMA_P_RANGE:
        for q in SARIMA_Q_RANGE:
            for sp in SARIMA_SP_RANGE:
                for sd in SARIMA_SD_RANGE:
                    for sq in SARIMA_SQ_RANGE:
                        try:
                            order = (p, suggested_d, q)
                            seasonal_order = (sp, sd, sq, SEASONAL_PERIOD)
                            fitted = SARIMAX(
                                train,
                                order=order,
                                seasonal_order=seasonal_order,
                                trend="n",
                                enforce_stationarity=False,
                                enforce_invertibility=False,
                            ).fit(disp=False)
                            forecast = fitted.get_forecast(steps=len(test)).predicted_mean
                            forecast.index = test.index
                            metrics = compute_metrics(test, forecast)
                            rows.append(
                                {
                                    "p": p,
                                    "d": suggested_d,
                                    "q": q,
                                    "P": sp,
                                    "D": sd,
                                    "Q": sq,
                                    "s": SEASONAL_PERIOD,
                                    "MAE": metrics["MAE"],
                                    "MSE": metrics["MSE"],
                                    "RMSE": metrics["RMSE"],
                                    "MAPE": metrics["MAPE"],
                                    "AIC": float(fitted.aic),
                                    "BIC": float(fitted.bic),
                                }
                            )
                        except Exception:
                            continue

    if not rows:
        raise RuntimeError("SARIMA grid search не дал ни одной корректной модели.")

    grid_df = pd.DataFrame(rows).sort_values(["MAE", "AIC", "BIC"]).reset_index(drop=True)
    best_row = grid_df.iloc[0]
    best_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    best_seasonal = (int(best_row["P"]), int(best_row["D"]), int(best_row["Q"]), int(best_row["s"]))

    fitted = SARIMAX(
        train,
        order=best_order,
        seasonal_order=best_seasonal,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test)).predicted_mean
    forecast.index = test.index
    metrics = compute_metrics(test, forecast)
    residuals = pd.Series(fitted.resid, index=train.index)
    lb = acorr_ljungbox(residuals, lags=[min(10, len(residuals) // 3)], return_df=True)["lb_pvalue"].iloc[0]
    fitted_params = {name: round(float(value), 6) for name, value in fitted.params.items()}

    model = ModelResult(
        name="SARIMA",
        family="SARIMA",
        rationale=(
            "Поскольку ряд месячный, дополнительно проверяется сезонная авторегрессионная модель с периодом 12. "
            "SARIMA нужна, если годовая сезонность улучшает качество прогноза по сравнению с обычной ARIMA."
        ),
        params_text=f"order = {best_order}, seasonal_order = {best_seasonal}",
        fitted_params={"order": best_order, "seasonal_order": best_seasonal, "coefficients": fitted_params},
        forecast=forecast,
        residuals=residuals,
        metrics=metrics,
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        ljung_box_pvalue=float(lb),
    )
    return grid_df, model


def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    model_results: list[ModelResult],
    figures_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(train.index, train.values, linewidth=2.2, color="#2e8b57", label="Обучающая выборка")
    ax.plot(test.index, test.values, linewidth=2.2, color="#c0392b", label="Факт")

    palette = ["#1f4e79", "#7d3c98", "#d77a1f", "#34495e", "#16a085"]
    for color, model in zip(palette, model_results):
        ax.plot(model.forecast.index, model.forecast.values, linewidth=2, linestyle="--", color=color, label=model.name)

    ax.set_title("Сравнение прогнозов моделей на тестовой выборке")
    ax.set_ylabel("Число запросов")
    ax.legend()
    ax.grid(True, alpha=0.25)

    path = figures_dir / "04_forecast_comparison.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def plot_residual_diagnostics(model: ModelResult, figures_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    residuals = model.residuals.dropna()

    axes[0, 0].plot(residuals.index, residuals.values, linewidth=1.5, color="#1f4e79")
    axes[0, 0].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[0, 0].set_title(f"Остатки: {model.name}")
    axes[0, 0].set_ylabel("Ошибка")

    sns.histplot(residuals, bins=20, kde=True, ax=axes[0, 1], color="#d77a1f")
    axes[0, 1].set_title("Гистограмма остатков")

    plot_acf(residuals, lags=min(20, len(residuals) // 2 - 1), ax=axes[1, 0])
    axes[1, 0].set_title("ACF остатков")

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q график")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"{model.name}: среднее остатков = {residuals.mean():.2f}, Ljung-Box p-value = {model.ljung_box_pvalue:.4f}",
        y=1.02,
    )

    path = figures_dir / f"05_residuals_{model.name}.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def plot_model_parameter_sensitivity(arima_grid_df: pd.DataFrame, sarima_grid_df: pd.DataFrame, figures_dir: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    flat_axes = axes.flatten()

    for ax, d in zip(flat_axes[: len(ARIMA_D_RANGE)], ARIMA_D_RANGE):
        subset = arima_grid_df[arima_grid_df["d"] == d]
        pivot = subset.pivot_table(index="p", columns="q", values="MAE", aggfunc="min")
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd_r", ax=ax)
        ax.set_title(f"MAE для ARIMA(p,{d},q)")
        ax.set_xlabel("q")
        ax.set_ylabel("p")

    sarima_ax = flat_axes[-1]
    sarima_top = sarima_grid_df.head(10).copy()
    sarima_top["config"] = sarima_top.apply(
        lambda row: f"({int(row['p'])},{int(row['d'])},{int(row['q'])}) x ({int(row['P'])},{int(row['D'])},{int(row['Q'])})",
        axis=1,
    )
    sns.barplot(data=sarima_top, x="MAE", y="config", ax=sarima_ax, color="#5a8f7b")
    sarima_ax.set_title("Топ-10 конфигураций SARIMA по MAE")
    sarima_ax.set_xlabel("MAE")
    sarima_ax.set_ylabel("Параметры")

    path = figures_dir / "06_parameter_sensitivity.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def forecast_with_model(train: pd.Series, horizon: int, model: ModelResult) -> pd.Series:
    if model.name == "ARIMA":
        order = tuple(model.fitted_params["order"])
        fitted = ARIMA(
            train,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast = fitted.get_forecast(steps=horizon).predicted_mean
    elif model.name == "SARIMA":
        order = tuple(model.fitted_params["order"])
        seasonal_order = tuple(model.fitted_params["seasonal_order"])
        fitted = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = fitted.get_forecast(steps=horizon).predicted_mean
    else:
        raise ValueError(f"Неизвестная модель для horizon analysis: {model.name}")
    return pd.Series(forecast).reset_index(drop=True)


def evaluate_horizon_dependency(ts: pd.Series, chosen_models: list[ModelResult]) -> pd.DataFrame:
    min_train = max(36, SEASONAL_PERIOD * 2)
    max_horizon = min(8, len(ts) - min_train - 1)
    rows = []

    for model in chosen_models:
        for horizon in range(1, max_horizon + 1):
            actual_values = []
            predicted_values = []
            for end in range(min_train, len(ts) - horizon + 1):
                train_slice = ts.iloc[:end]
                actual = ts.iloc[end + horizon - 1]
                forecast = forecast_with_model(train_slice, horizon, model)
                actual_values.append(actual)
                predicted_values.append(float(forecast.iloc[horizon - 1]))

            actual_series = pd.Series(actual_values)
            predicted_series = pd.Series(predicted_values, index=actual_series.index)
            metrics = compute_metrics(actual_series, predicted_series)
            rows.append(
                {
                    "model": model.name,
                    "horizon": horizon,
                    "MAE": metrics["MAE"],
                    "MSE": metrics["MSE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE": metrics["MAPE"],
                }
            )

    return pd.DataFrame(rows)


def plot_horizon_dependency(horizon_df: pd.DataFrame, figures_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["MAE", "RMSE", "MAPE"]

    for ax, metric in zip(axes, metrics):
        for model_name, subset in horizon_df.groupby("model"):
            ax.plot(subset["horizon"], subset[metric], marker="o", linewidth=2, label=model_name)
        ax.set_title(f"{metric} по горизонтам прогноза")
        ax.set_xlabel("Горизонт, шагов")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    axes[0].legend()
    path = figures_dir / "07_horizon_analysis.png"
    saved_path = save_figure(fig, path)
    return saved_path.name


def round_float(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def model_results_to_frame(model_results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for model in model_results:
        rows.append(
            {
                "Модель": model.name,
                "Семейство": model.family,
                "MAE": model.metrics["MAE"],
                "MSE": model.metrics["MSE"],
                "RMSE": model.metrics["RMSE"],
                "MAPE, %": model.metrics["MAPE"],
                "AIC": model.aic,
                "BIC": model.bic,
                "Ljung-Box p-value": model.ljung_box_pvalue,
                "Параметры": model.params_text,
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def console_table(df: pd.DataFrame, decimals: int = 2) -> str:
    formatters = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            formatters[col] = lambda value: f"{int(value)}"
        elif pd.api.types.is_numeric_dtype(df[col]):
            formatters[col] = lambda value, d=decimals: f"{value:,.{d}f}"
    return df.to_string(
        index=False,
        formatters=formatters,
    )


def main() -> None:
    figures_dir = ensure_figures_dir()

    print_section(1, "Загрузка и подготовка данных")
    ts, date_col, value_col, missing_filled = load_series(DATA_FILE)
    train, test = train_test_split_series(ts)
    print(f"Файл данных: {DATA_FILE.name}")
    print(f"Столбец даты: {date_col}")
    print(f"Столбец значений: {value_col}")
    print(f"Наблюдений: {len(ts)}")
    print(f"Период: {ts.index.min().strftime('%m.%Y')} - {ts.index.max().strftime('%m.%Y')}")
    print(f"Обучающая выборка: {len(train)}, тестовая выборка: {len(test)}")
    print(f"Пропуски, заполненные интерполяцией: {missing_filled}")

    print_section(2, "Визуальный анализ")
    visual_figure = plot_visual_analysis(ts, train, test, figures_dir)
    print(f"График сохранен: {figures_dir / visual_figure}")

    print_section(3, "Оценка выбросов по критерию MAD")
    outlier_mask, modified_z, median_value, mad_value = detect_outliers_mad(ts)
    outlier_figure = plot_outliers(ts, outlier_mask, modified_z, figures_dir)
    print(f"Медиана: {median_value:.2f}")
    print(f"MAD: {mad_value:.2f}")
    print(f"Выбросов по критерию |modified z| > 3.5: {int(outlier_mask.sum())}")
    if outlier_mask.any():
        sample_outliers = ts[outlier_mask].head(10)
        for idx, value in sample_outliers.items():
            print(f"  {idx.strftime('%m.%Y')}: {value:.0f}")
    print(f"График сохранен: {figures_dir / outlier_figure}")

    print_section(4, "Стационарность и коррелограммы")
    stationarity = stationarity_summary(ts)
    acf_figure, acf_analysis = plot_acf_pacf_overview(ts, figures_dir)
    print(f"ADF p-value (исходный ряд): {stationarity['original']['ADF_p']:.6f}")
    print(f"KPSS p-value (исходный ряд): {stationarity['original']['KPSS_p']:.6f}")
    print(f"ADF p-value (1-я разность): {stationarity['diff1']['ADF_p']:.6f}")
    print(f"KPSS p-value (1-я разность): {stationarity['diff1']['KPSS_p']:.6f}")
    print(stationarity["conclusion"])
    print(f"Значимые лаги ACF первой разности: {acf_analysis['diff_acf_lags']}")
    print(f"Значимые лаги PACF первой разности: {acf_analysis['diff_pacf_lags']}")
    print(f"График сохранен: {figures_dir / acf_figure}")

    print_section(5, "Обучение моделей ARIMA и SARIMA")
    arima_grid_df, arima_model = run_arima_grid_search(train, test, stationarity["suggested_d"])
    sarima_grid_df, sarima_model = run_sarima_grid_search(train, test, stationarity["suggested_d"])
    model_results = [arima_model, sarima_model]

    for model in model_results:
        print(
            f"{model.name:<14} MAE={model.metrics['MAE']:.2f}  "
            f"MSE={model.metrics['MSE']:.2f}  RMSE={model.metrics['RMSE']:.2f}  "
            f"MAPE={model.metrics['MAPE']:.2f}%"
        )
        print(f"  Параметры: {model.params_text}")

    metrics_df = model_results_to_frame(model_results)
    comparison_figure = plot_forecast_comparison(train, test, model_results, figures_dir)
    print(f"График прогнозов сохранен: {figures_dir / comparison_figure}")

    print_section(6, "Диагностика остатков")
    for model in model_results:
        model.diagnostic_figure = plot_residual_diagnostics(model, figures_dir)
        print(
            f"{model.name:<14} mean(resid)={model.residuals.mean():.2f}, "
            f"Ljung-Box p-value={model.ljung_box_pvalue:.4f}"
        )

    print_section(7, "Исследование влияния параметров ARIMA и SARIMA")
    sensitivity_figure = plot_model_parameter_sensitivity(arima_grid_df, sarima_grid_df, figures_dir)
    best_arima_row = arima_grid_df.iloc[0]
    best_sarima_row = sarima_grid_df.iloc[0]
    print(
        f"Лучшая ARIMA по MAE: ({int(best_arima_row['p'])},{int(best_arima_row['d'])},{int(best_arima_row['q'])}) "
        f"с MAE={best_arima_row['MAE']:.2f}"
    )
    print(
        "Лучшая SARIMA по MAE: "
        f"({int(best_sarima_row['p'])},{int(best_sarima_row['d'])},{int(best_sarima_row['q'])}) x "
        f"({int(best_sarima_row['P'])},{int(best_sarima_row['D'])},{int(best_sarima_row['Q'])},{int(best_sarima_row['s'])}) "
        f"с MAE={best_sarima_row['MAE']:.2f}"
    )
    print(f"График чувствительности сохранен: {figures_dir / sensitivity_figure}")

    print_section(8, "Зависимость точности от горизонта прогноза")
    horizon_df = evaluate_horizon_dependency(ts, model_results)
    horizon_df = horizon_df.sort_values(["model", "horizon"]).reset_index(drop=True)
    horizon_figure = plot_horizon_dependency(horizon_df, figures_dir)
    print(console_table(horizon_df[["model", "horizon", "MAE", "RMSE", "MAPE"]], decimals=2))
    print(f"График горизонтов сохранен: {figures_dir / horizon_figure}")

    print_section(9, "Краткий итог")
    print(f"Лучшая модель по MAE: {metrics_df.iloc[0]['Модель']}")
    print(f"MAE = {metrics_df.iloc[0]['MAE']:.2f}")
    print(f"MAPE = {metrics_df.iloc[0]['MAPE, %']:.2f}%")
    print(f"Графики сохранены в: {figures_dir}")


if __name__ == "__main__":
    main()
