# RGR_TSP_

Проект для анализа временного ряда запросов о смертности в России.

## Что делает `quick_analysis.py`

- загружает ряд из `wordstat_dynamic.xlsx`
- выполняет визуальный анализ
- ищет выбросы по критерию MAD
- проверяет стационарность через ADF и KPSS
- строит ACF и PACF
- сравнивает только модели `ARIMA` и `SARIMA`
- считает `MAE`, `MSE`, `RMSE`, `MAPE`
- строит диагностику остатков
- исследует влияние параметров моделей
- исследует изменение точности по горизонтам прогноза

## Выходные файлы

Все графики сохраняются прямо в папку `figures/`.

Основные файлы:

- `01_visual_analysis.png`
- `02_outlier_analysis.png`
- `03_acf_pacf.png`
- `04_forecast_comparison.png`
- `05_residuals_ARIMA.png`
- `05_residuals_SARIMA.png`
- `06_parameter_sensitivity.png`
- `07_horizon_analysis.png`

## Запуск

```bash
.venv\Scripts\python.exe quick_analysis.py
```

## Зависимости

См. `requirements.txt`.
