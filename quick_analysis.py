import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy import stats

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

print("\n(1)  ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("-" * 80)

try:
    df = pd.read_excel('wordstat_dynamic.xlsx')
    print(f"✓ Данные успешно загружены. Размер: {df.shape}")
    print(f"\nПервые 5 строк:\n{df.head()}")
except Exception as e:
    print(f"✗ Ошибка при загрузке данных: {e}")
    exit(1)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) == 0:
    print("Нет числовых столбцов")
    exit(1)

value_col = numeric_cols[0]
ts = pd.Series(df[value_col].values)
ts = ts.dropna()

split_point = int(len(ts) * 0.8)
train_data = ts[:split_point]
test_data = ts[split_point:]

print(f"\ Временной ряд подготовлен:")
print(f"  • Всего наблюдений: {len(ts)}")
print(f"  • Обучающая выборка: {len(train_data)} ({len(train_data)/len(ts)*100:.1f}%)")
print(f"  • Тестовая выборка: {len(test_data)} ({len(test_data)/len(ts)*100:.1f}%)")
print(f"  • Минимум: {ts.min():.4f}")
print(f"  • Максимум: {ts.max():.4f}")
print(f"  • Среднее: {ts.mean():.4f}")
print(f"  • Стд. отклонение: {ts.std():.4f}")

print("\n(2)  ВИЗУАЛИЗАЦИЯ ВРЕМЕННОГО РЯДА")
print("-" * 80)

os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(ts.values, linewidth=2, label='Исходный ряд', color='#1f77b4')
axes[0].fill_between(range(len(ts)), ts.values, alpha=0.3, color='#1f77b4')
axes[0].set_title('Исходный временной ряд', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Значение')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(len(train_data)), train_data.values, label='Обучающая выборка', 
             linewidth=2, color='#2ca02c')
axes[1].plot(range(len(train_data), len(ts)), test_data.values, label='Тестовая выборка', 
             linewidth=2, color='#d62728')
axes[1].set_title('Разделение на обучающую и тестовую выборки', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Значение')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].hist(ts.values, bins=30, edgecolor='black', alpha=0.7, color='#ff7f0e')
axes[2].set_title('Распределение значений', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Значение')
axes[2].set_ylabel('Частота')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/01_time_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("График временного ряда сохранен: figures/01_time_series.png")

print("\n(3)  АНАЛИЗ ВЫБРОСОВ")
print("-" * 80)

# Метод IQR
Q1 = ts.quantile(0.25)
Q3 = ts.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ts[(ts < Q1 - 1.5*IQR) | (ts > Q3 + 1.5*IQR)]

# Метод MAD
median = ts.median()
mad = np.median(np.abs(ts - median))
modified_z = 0.6745 * (ts - median) / mad if mad != 0 else 0
outliers_mad = ts[np.abs(modified_z) > 2.5]

# Метод Z-score
z_scores = np.abs(stats.zscore(ts))
outliers_zscore = ts[z_scores > 3]

print(f"\n Результаты анализа выбросов:")
print(f"  • Метод IQR: {len(outliers_iqr)} выбросов")
print(f"  • Метод MAD: {len(outliers_mad)} выбросов")
print(f"  • Метод Z-score: {len(outliers_zscore)} выбросов")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(ts.values, linewidth=2, label='Ряд', color='#1f77b4')
axes[0, 0].scatter(outliers_iqr.index, outliers_iqr.values, color='red', s=100, 
                   label=f'Выбросы ({len(outliers_iqr)})', zorder=5)
axes[0, 0].set_title('Метод IQR', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(ts.values, linewidth=2, label='Ряд', color='#1f77b4')
axes[0, 1].scatter(outliers_mad.index, outliers_mad.values, color='red', s=100, 
                   label=f'Выбросы ({len(outliers_mad)})', zorder=5)
axes[0, 1].set_title('Метод MAD', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(ts.values, linewidth=2, label='Ряд', color='#1f77b4')
axes[1, 0].scatter(outliers_zscore.index, outliers_zscore.values, color='red', s=100, 
                   label=f'Выбросы ({len(outliers_zscore)})', zorder=5)
axes[1, 0].set_title('Метод Z-score', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

all_outliers_idx = list(set(list(outliers_iqr.index) + list(outliers_mad.index) + 
                            list(outliers_zscore.index)))
axes[1, 1].plot(ts.values, linewidth=2, label='Ряд', color='#1f77b4')
if len(all_outliers_idx) > 0:
    axes[1, 1].scatter(all_outliers_idx, ts.iloc[all_outliers_idx].values, 
                       color='red', s=100, label=f'Выбросы ({len(all_outliers_idx)})', zorder=5)
axes[1, 1].set_title('Объединение всех методов', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/02_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Анализ выбросов сохранен: figures/02_outliers.png")

print("\n(4)  АНАЛИЗ СТАЦИОНАРНОСТИ")
print("-" * 80)

adf_result = adfuller(ts)
print(f"\n Тест Дики-Фуллера (ADF):")
print(f"  • ADF статистика: {adf_result[0]:.6f}")
print(f"  • p-value: {adf_result[1]:.6f}")
print(f"  • Критическое значение (5%): {adf_result[4]['5%']:.3f}")
if adf_result[1] < 0.05:
    print(f"  Ряд стационарен")
else:
    print(f"  Ряд нестационарен")

kpss_result = kpss(ts, regression='c')
print(f"\n Тест KPSS:")
print(f"  • KPSS статистика: {kpss_result[0]:.6f}")
print(f"  • p-value: {kpss_result[1]:.6f}")
if kpss_result[1] > 0.05:
    print(f"   Ряд стационарен")
else:
    print(f"   Ряд нестационарен")

print("\n(5)  ПОСТРОЕНИЕ КОРРЕЛОГРАММ (АКФ и ЧАКФ)")
print("-" * 80)

max_lags = min(20, len(ts) // 2 - 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

plot_acf(ts, lags=max_lags, ax=axes[0, 0])
axes[0, 0].set_title('АКФ (Автокорреляционная функция)', fontweight='bold')

plot_pacf(ts, lags=max_lags, ax=axes[0, 1])
axes[0, 1].set_title('ЧАКФ (Частная автокорреляционная функция)', fontweight='bold')

diff_ts = ts.diff().dropna()
plot_acf(diff_ts, lags=max_lags, ax=axes[1, 0])
axes[1, 0].set_title('АКФ первых разностей', fontweight='bold')

plot_pacf(diff_ts, lags=min(max_lags, len(diff_ts) // 2 - 1), ax=axes[1, 1])
axes[1, 1].set_title('ЧАКФ первых разностей', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/03_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Коррелограммы сохранены: figures/03_acf_pacf.png")

print("\n(6)  ОБУЧЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
print("-" * 80)

models = {}
forecasts = {}
metrics = {}

# Модель 1: ARIMA
print("\n▪ ARIMA(1,1,1)")
try:
    arima_model = ARIMA(train_data, order=(1, 1, 1))
    arima_results = arima_model.fit()
    arima_forecast = arima_results.get_forecast(steps=len(test_data)).predicted_mean
    
    mae_arima = mean_absolute_error(test_data, arima_forecast)
    mse_arima = mean_squared_error(test_data, arima_forecast)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = mean_absolute_percentage_error(test_data, arima_forecast)
    
    models['ARIMA'] = arima_results
    forecasts['ARIMA'] = arima_forecast
    metrics['ARIMA'] = {'MAE': mae_arima, 'MSE': mse_arima, 'RMSE': rmse_arima, 'MAPE': mape_arima}
    
    print(f"  MAE={mae_arima:.4f}, RMSE={rmse_arima:.4f}, MAPE={mape_arima:.4f}%")
except Exception as e:
    print(f"  Ошибка: {e}")

# Модель 2: Exponential Smoothing
print("\n▪ Exponential Smoothing (Holt-Winters)")
try:
    if len(train_data) >= 24:
        try:
            exp_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add')
            exp_results = exp_model.fit()
        except:
            print("  Использую модель Холта (двухпараметрическое сглаживание)")
            from statsmodels.tsa.holtwinters import Holt
            exp_model = Holt(train_data)
            exp_results = exp_model.fit()
    else:
        print("  Недостаточно наблюдений, используем простое экспоненциальное сглаживание")
        exp_model = SimpleExpSmoothing(train_data)
        exp_results = exp_model.fit()
    
    exp_forecast = exp_results.forecast(steps=len(test_data))
    
    mae_exp = mean_absolute_error(test_data, exp_forecast)
    mse_exp = mean_squared_error(test_data, exp_forecast)
    rmse_exp = np.sqrt(mse_exp)
    mape_exp = mean_absolute_percentage_error(test_data, exp_forecast)
    
    models['ExpSmoothing'] = exp_results
    forecasts['ExpSmoothing'] = exp_forecast
    metrics['ExpSmoothing'] = {'MAE': mae_exp, 'MSE': mse_exp, 'RMSE': rmse_exp, 'MAPE': mape_exp}
    
    print(f"  MAE={mae_exp:.4f}, RMSE={rmse_exp:.4f}, MAPE={mape_exp:.4f}%")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n(7)  СРАВНЕНИЕ ПРОГНОЗОВ")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(range(len(train_data)), train_data.values, label='Обучающая выборка', 
        linewidth=2, color='#2ca02c')
ax.plot(range(len(train_data), len(ts)), test_data.values, label='Фактические значения', 
        linewidth=2, color='#d62728')

colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#8c564b']
for (model_name, forecast), color in zip(forecasts.items(), colors):
    ax.plot(range(len(train_data), len(ts)), forecast.values, label=f'Прогноз {model_name}', 
            linewidth=2, linestyle='--', color=color, alpha=0.7)

ax.set_title('Сравнение прогнозов моделей', fontsize=14, fontweight='bold')
ax.set_xlabel('Индекс наблюдения')
ax.set_ylabel('Значение')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/04_forecast_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("График сравнения прогнозов сохранен: figures/04_forecast_comparison.png")

print("\n(8)  АНАЛИЗ ОСТАТКОВ")
print("-" * 80)

for model_name, model_result in models.items():
    residuals = model_result.resid
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(residuals.values, linewidth=1, color='#1f77b4')
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('График остатков', fontweight='bold')
    axes[0, 0].set_ylabel('Остатки')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(residuals.values, bins=20, edgecolor='black', alpha=0.7, color='#ff7f0e')
    axes[0, 1].set_title('Распределение остатков', fontweight='bold')
    axes[0, 1].set_xlabel('Остатки')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    plot_acf(residuals, lags=20, ax=axes[1, 0])
    axes[1, 0].set_title('АКФ остатков', fontweight='bold')

    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q диаграмма', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'figures/05_residuals_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Анализ остатков {model_name}: figures/05_residuals_{model_name}.png")


print("\n(9) ТАБЛИЦА МЕТРИК:")
print("-" * 80)
print(f"{'Модель':<20} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12}")
print("-" * 80)
for model_name, metric_dict in metrics.items():
    print(f"{model_name:<20} {metric_dict['MAE']:<12.4f} {metric_dict['RMSE']:<12.4f} {metric_dict['MAPE']:<12.4f}")

if metrics:
    best_model = min(metrics.items(), key=lambda x: x[1]['MAE'])
    print("-" * 80)
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model[0]} (MAE = {best_model[1]['MAE']:.4f})")

# ============================================================================
# 10. ГЕНЕРАЦИЯ HTML ОТЧЕТА
# ============================================================================
print("\n(10) ГЕНЕРАЦИЯ HTML ОТЧЕТА")
print("-" * 80)

html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>РГР: Анализ временных рядов</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        table th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
        table td {{ border: 1px solid #ddd; padding: 10px; }}
        table tr:nth-child(even) {{ background: #f9f9f9; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; }}
        .stats {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .best {{ background: #d5f4e6; border-left: 4px solid #27ae60; padding: 15px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>РАСЧЕТНО-ГРАФИЧЕСКАЯ РАБОТА</h1>
        <h1 style="font-size: 20px; border: none;">Анализ временных рядов</h1>
        
        <div class="stats">
            <p><strong>Количество наблюдений:</strong> {len(ts)}</p>
            <p><strong>Минимум:</strong> {ts.min():.4f} | <strong>Максимум:</strong> {ts.max():.4f}</p>
            <p><strong>Среднее:</strong> {ts.mean():.4f} | <strong>Стд. откл.:</strong> {ts.std():.4f}</p>
        </div>
        
        <h2>1. Визуализация временного ряда</h2>
        <img src="figures/01_time_series.png" alt="Временной ряд">
        
        <h2>2. Анализ выбросов</h2>
        <img src="figures/02_outliers.png" alt="Выбросы">
        <p>IQR: {len(outliers_iqr)} выбросов | MAD: {len(outliers_mad)} выбросов | Z-score: {len(outliers_zscore)} выбросов</p>
        
        <h2>3. Коррелограммы (АКФ и ЧАКФ)</h2>
        <img src="figures/03_acf_pacf.png" alt="Коррелограммы">
        
        <h2>4. Прогнозы моделей</h2>
        <img src="figures/04_forecast_comparison.png" alt="Прогнозы">
        
        <h2>5. Метрики качества моделей</h2>
        <table>
            <tr>
                <th>Модель</th>
                <th>MAE</th>
                <th>RMSE</th>
                <th>MAPE (%)</th>
            </tr>
"""

for model_name, metric_dict in metrics.items():
    html_content += f"""
            <tr>
                <td><strong>{model_name}</strong></td>
                <td>{metric_dict['MAE']:.4f}</td>
                <td>{metric_dict['RMSE']:.4f}</td>
                <td>{metric_dict['MAPE']:.4f}</td>
            </tr>
"""

html_content += """
        </table>
        
        <h2>6. Анализ остатков</h2>
"""

for model_name in models.keys():
    html_content += f"""
        <h3>Модель {model_name}</h3>
        <img src="figures/05_residuals_{model_name}.png" alt="Остатки {model_name}">
"""

if best_model:
    html_content += f"""
        <div class="best">
            <h2 style="border: none; padding: 0; margin: 0;">🏆 ЛУЧШАЯ МОДЕЛЬ</h2>
            <p><strong>{best_model[0]}</strong> с MAE = {best_model[1]['MAE']:.4f}</p>
        </div>
"""

html_content += """
        <h2>7. Выводы</h2>
        <p>На основе проведенного анализа:</p>
        <ul>
            <li>Выполнена визуализация и анализ выбросов в данных</li>
            <li>Построены коррелограммы для выявления автокорреляции</li>
            <li>Обучены две модели прогнозирования</li>
            <li>Оценена качество каждой модели</li>
            <li>Проведен анализ остатков</li>
        </ul>
    </div>
</body>
</html>
"""

with open('RGR_Report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✓ HTML отчет сохранен: RGR_Report.html")
