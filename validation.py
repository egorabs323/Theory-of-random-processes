from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class AnalysisConfig:
    data_file: Path
    figures_dir: Path
    test_share: float
    outlier_threshold: float
    seasonal_period: int
    arima_p_max: int
    arima_d_max: int
    arima_q_max: int
    sarima_p_max: int
    sarima_q_max: int
    sarima_P_max: int
    sarima_D_max: int
    sarima_Q_max: int
    acf_max_lags: int
    horizon_max: int
    min_train_size: int | None

    @property
    def arima_p_range(self) -> range:
        return range(0, self.arima_p_max + 1)

    @property
    def arima_d_range(self) -> range:
        return range(0, self.arima_d_max + 1)

    @property
    def arima_q_range(self) -> range:
        return range(0, self.arima_q_max + 1)

    @property
    def sarima_p_range(self) -> range:
        return range(0, self.sarima_p_max + 1)

    @property
    def sarima_q_range(self) -> range:
        return range(0, self.sarima_q_max + 1)

    @property
    def sarima_P_range(self) -> range:
        return range(0, self.sarima_P_max + 1)

    @property
    def sarima_D_range(self) -> range:
        return range(0, self.sarima_D_max + 1)

    @property
    def sarima_Q_range(self) -> range:
        return range(0, self.sarima_Q_max + 1)


def print_section(number: int, title: str) -> None:
    print(f"\n({number}) {title}")
    print("-" * 90)


def default_config(base_dir: Path) -> AnalysisConfig:
    seasonal_period = 12
    return AnalysisConfig(
        data_file=base_dir / "wordstat_dynamic.xlsx",
        figures_dir=base_dir / "figures",
        test_share=0.2,
        outlier_threshold=3.5,
        seasonal_period=seasonal_period,
        arima_p_max=3,
        arima_d_max=2,
        arima_q_max=3,
        sarima_p_max=2,
        sarima_q_max=2,
        sarima_P_max=1,
        sarima_D_max=1,
        sarima_Q_max=1,
        acf_max_lags=24,
        horizon_max=8,
        min_train_size=None,
    )


def validate_config(config: AnalysisConfig) -> AnalysisConfig:
    if not config.data_file.exists() or not config.data_file.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {config.data_file}")
    if config.data_file.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError("Файл данных должен быть Excel-файлом .xlsx или .xls.")

    if config.figures_dir.exists() and not config.figures_dir.is_dir():
        raise ValueError(f"Путь для графиков должен указывать на папку: {config.figures_dir}")

    if not 0.05 <= config.test_share <= 0.5:
        raise ValueError("Доля тестовой выборки должна быть в диапазоне от 0.05 до 0.5.")
    if config.outlier_threshold <= 0:
        raise ValueError("Порог выбросов должен быть положительным.")
    if config.seasonal_period < 1:
        raise ValueError("Сезонный период должен быть не меньше 1.")
    if config.arima_p_max < 0 or config.arima_d_max < 0 or config.arima_q_max < 0:
        raise ValueError("Максимумы p, d, q для ARIMA не могут быть отрицательными.")
    if config.sarima_p_max < 0 or config.sarima_q_max < 0:
        raise ValueError("Максимумы p и q для SARIMA не могут быть отрицательными.")
    if config.sarima_P_max < 0 or config.sarima_D_max < 0 or config.sarima_Q_max < 0:
        raise ValueError("Максимумы P, D, Q для SARIMA не могут быть отрицательными.")
    if config.acf_max_lags < 5:
        raise ValueError("Максимум лагов для ACF/PACF должен быть не меньше 5.")
    if config.horizon_max < 1:
        raise ValueError("Максимальный горизонт прогноза должен быть не меньше 1.")
    if config.min_train_size is not None and config.min_train_size < 12:
        raise ValueError("Минимальный размер обучающей выборки должен быть не меньше 12.")
    return config


def prompt_text(label: str, default: str) -> str:
    answer = input(f"{label} [{default}]: ").strip()
    return answer or default


def prompt_int(label: str, default: int, min_value: int | None = None) -> int:
    while True:
        answer = input(f"{label} [{default}]: ").strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            print("Введите целое число.")
            continue
        if min_value is not None and value < min_value:
            print(f"Значение должно быть не меньше {min_value}.")
            continue
        return value


def prompt_float(label: str, default: float, min_value: float | None = None, max_value: float | None = None) -> float:
    while True:
        answer = input(f"{label} [{default}]: ").strip().replace(",", ".")
        if not answer:
            return default
        try:
            value = float(answer)
        except ValueError:
            print("Введите число.")
            continue
        if min_value is not None and value < min_value:
            print(f"Значение должно быть не меньше {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Значение должно быть не больше {max_value}.")
            continue
        return value


def prompt_optional_int(label: str, default_text: str, min_value: int | None = None) -> int | None:
    while True:
        answer = input(f"{label} [{default_text}]: ").strip()
        if not answer:
            return None
        try:
            value = int(answer)
        except ValueError:
            print("Введите целое число или оставьте поле пустым.")
            continue
        if min_value is not None and value < min_value:
            print(f"Значение должно быть не меньше {min_value}.")
            continue
        return value


def resolve_local_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def prompt_config(base_dir: Path) -> AnalysisConfig:
    config = default_config(base_dir)
    print_section(0, "Настройка параметров")
    print("Нажмите Enter, чтобы оставить значение по умолчанию.")

    try:
        data_file = resolve_local_path(base_dir, prompt_text("Файл данных Excel", str(config.data_file.name)))
        figures_dir = resolve_local_path(base_dir, prompt_text("Папка для графиков", str(config.figures_dir.name)))
        test_share = prompt_float("Доля тестовой выборки", config.test_share, min_value=0.05, max_value=0.5)
        outlier_threshold = prompt_float("Порог выбросов |modified z|", config.outlier_threshold, min_value=0.1)
        seasonal_period = prompt_int("Сезонный период SARIMA", config.seasonal_period, min_value=1)
        arima_p_max = prompt_int("Максимум p для ARIMA", config.arima_p_max, min_value=0)
        arima_d_max = prompt_int("Максимум d для ARIMA", config.arima_d_max, min_value=0)
        arima_q_max = prompt_int("Максимум q для ARIMA", config.arima_q_max, min_value=0)
        sarima_p_max = prompt_int("Максимум p для SARIMA", config.sarima_p_max, min_value=0)
        sarima_q_max = prompt_int("Максимум q для SARIMA", config.sarima_q_max, min_value=0)
        sarima_P_max = prompt_int("Максимум P для SARIMA", config.sarima_P_max, min_value=0)
        sarima_D_max = prompt_int("Максимум D для SARIMA", config.sarima_D_max, min_value=0)
        sarima_Q_max = prompt_int("Максимум Q для SARIMA", config.sarima_Q_max, min_value=0)
        acf_max_lags = prompt_int("Максимум лагов для ACF/PACF", config.acf_max_lags, min_value=5)
        horizon_max = prompt_int("Максимальный горизонт прогноза", config.horizon_max, min_value=1)
        auto_min_train = max(36, seasonal_period * 2)
        min_train_size = prompt_optional_int(
            "Минимум обучающих наблюдений для анализа горизонтов",
            f"auto = {auto_min_train}",
            min_value=12,
        )
    except EOFError:
        print("Ввод параметров недоступен, используются параметры по умолчанию.")
        return validate_config(config)

    return validate_config(
        AnalysisConfig(
            data_file=data_file,
            figures_dir=figures_dir,
            test_share=test_share,
            outlier_threshold=outlier_threshold,
            seasonal_period=seasonal_period,
            arima_p_max=arima_p_max,
            arima_d_max=arima_d_max,
            arima_q_max=arima_q_max,
            sarima_p_max=sarima_p_max,
            sarima_q_max=sarima_q_max,
            sarima_P_max=sarima_P_max,
            sarima_D_max=sarima_D_max,
            sarima_Q_max=sarima_Q_max,
            acf_max_lags=acf_max_lags,
            horizon_max=horizon_max,
            min_train_size=min_train_size,
        )
    )


def ensure_figures_dir(figures_dir: Path) -> Path:
    if figures_dir.exists() and not figures_dir.is_dir():
        raise ValueError(f"Невозможно создать папку для графиков: {figures_dir}")
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


def validate_source_dataframe(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        raise ValueError(f"Файл данных пуст: {path}")
    if len(df.columns) < 2:
        raise ValueError("В Excel-файле должно быть минимум два столбца: дата и числовые значения.")


def validate_series(ts: pd.Series, seasonal_period: int) -> pd.Series:
    if ts.empty:
        raise ValueError("После очистки не осталось наблюдений для анализа.")
    if len(ts) < max(12, seasonal_period + 2):
        raise ValueError(
            f"Для анализа нужен ряд минимум из {max(12, seasonal_period + 2)} наблюдений, сейчас доступно {len(ts)}."
        )
    if ts.nunique(dropna=True) < 2:
        raise ValueError("Временной ряд почти константный, модели ARIMA/SARIMA на нем неинформативны.")
    if not ts.index.is_monotonic_increasing:
        raise ValueError("Временной индекс должен быть отсортирован по возрастанию.")
    if ts.isna().any():
        raise ValueError("Во временном ряду остались пропуски после подготовки данных.")
    return ts


def load_series(path: Path, seasonal_period: int) -> tuple[pd.Series, str, str, int]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {path}")

    df = pd.read_excel(path)
    validate_source_dataframe(df, path)
    date_col = choose_date_column(df)
    value_col = choose_value_column(df, date_col)

    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna().sort_values(date_col).drop_duplicates(subset=date_col)
    if work.empty:
        raise ValueError("После преобразования дат и значений не осталось корректных строк.")

    ts = work.set_index(date_col)[value_col].astype(float).asfreq("MS")
    missing_before_fill = int(ts.isna().sum())
    if missing_before_fill:
        ts = ts.interpolate(method="time").ffill().bfill()

    ts.name = value_col
    return validate_series(ts, seasonal_period), date_col, value_col, missing_before_fill


def train_test_split_series(ts: pd.Series, test_share: float = 0.2) -> tuple[pd.Series, pd.Series]:
    split_idx = int(len(ts) * (1 - test_share))
    if split_idx <= 0 or split_idx >= len(ts):
        raise ValueError("Разбиение train/test дало пустую обучающую или тестовую выборку.")
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    if len(train) < 8:
        raise ValueError("После разбиения обучающая выборка слишком мала для оценки ARIMA/SARIMA.")
    if len(test) < 1:
        raise ValueError("Тестовая выборка не должна быть пустой.")
    return train, test


def resolve_acf_pacf_lags(ts: pd.Series, requested_max_lags: int) -> dict[str, int]:
    diff1 = ts.diff().dropna()
    if len(diff1) < 3:
        raise ValueError("Для построения ACF/PACF после разностного преобразования недостаточно наблюдений.")

    max_lags = min(requested_max_lags, max(8, len(ts) // 3))
    max_lags = min(max_lags, len(ts) - 1, len(diff1) - 1)
    pacf_original_lags = min(max_lags, len(ts) // 2 - 1)
    pacf_diff_lags = min(max_lags, len(diff1) // 2 - 1)

    if max_lags < 1 or pacf_original_lags < 1 or pacf_diff_lags < 1:
        raise ValueError("Недостаточно наблюдений для корректного построения ACF/PACF.")

    return {
        "max_lags": max_lags,
        "pacf_original_lags": pacf_original_lags,
        "pacf_diff_lags": pacf_diff_lags,
    }


def validate_grid_search_rows(rows: list[dict], model_name: str) -> None:
    if rows:
        return
    raise RuntimeError(f"{model_name} grid search не дал ни одной корректной модели.")


def validate_supported_model_name(model_name: str) -> None:
    if model_name in {"ARIMA", "SARIMA"}:
        return
    raise ValueError(f"Неизвестная модель для horizon analysis: {model_name}")


def resolve_horizon_settings(ts_length: int, config: AnalysisConfig) -> tuple[int, int]:
    auto_min_train = max(36, config.seasonal_period * 2)
    min_train = config.min_train_size if config.min_train_size is not None else auto_min_train
    min_train = min(min_train, max(ts_length - 2, 1))
    max_horizon = min(config.horizon_max, ts_length - min_train - 1)
    if max_horizon < 1:
        raise ValueError("Недостаточно наблюдений для анализа горизонтов при выбранных параметрах.")
    return min_train, max_horizon
