# predict_demand.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# В начале файла добавьте проверку размера данных
df = pd.read_csv('canteen_sales_dataset.csv', parse_dates=['date'], encoding='cp1251')
print(f"Загружено {len(df)} записей из базы данных")

# Если данных много, можно взять только последние N дней для ускорения
if len(df) > 5000:
    print("Данных много, беру последние 180 дней для обучения...")
    cutoff_date = df['date'].max() - timedelta(days=180)
    df = df[df['date'] >= cutoff_date]
    print(f"Оставлено {len(df)} записей для обучения")

print("=" * 60)
print("ШАГ 1: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ДЛЯ ПРОГНОЗА СПРОСА")
print("=" * 60)

# 1. Загружаем наш датасет
df = pd.read_csv('canteen_sales_dataset.csv', parse_dates=['date'])
print(f"✓ Загружен датасет: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 2. Создаем новые признаки для модели (это ключевой этап!)
# Из даты извлекаем дополнительные временные характеристики
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Создаем признак "прошлый спрос" (спрос в предыдущий день для того же блюда)
df['prev_day_demand'] = df.groupby('dish_id')['demand'].shift(1)

# Заполняем пропуски в прошлом спросе средним значением
df['prev_day_demand'].fillna(df['demand'].mean(), inplace=True)

print("✓ Созданы новые временные признаки для улучшения прогноза")

# 3. Подготовка данных для обучения
# Выбираем признаки (X) и целевую переменную (y)
feature_columns = [
    'day_of_week', 'month', 'day_of_month', 'is_month_start', 'is_month_end',
    'is_weekend', 'is_session', 'is_holiday', 'temperature',
    'base_cost', 'selling_price', 'prev_day_demand', 'dish_id'
]

X = df[feature_columns]  # Признаки
y = df['demand']         # Что предсказываем (спрос)

# 4. Разделяем данные на обучающую (80%) и тестовую (20%) выборки
# random_state=42 гарантирует воспроизводимость результата
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"✓ Данные разделены:")
print(f"  Обучающая выборка: {X_train.shape[0]} записей")
print(f"  Тестовая выборка: {X_test.shape[0]} записей")

print("\n" + "=" * 60)
print("ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ RANDOM FOREST ДЛЯ ПРОГНОЗА")
print("=" * 60)

# 5. Создаем и обучаем модель Random Forest
# Это ансамблевая модель, которая хорошо работает с табличными данными
model = RandomForestRegressor(
    n_estimators=100,   # Количество деревьев
    max_depth=10,       # Максимальная глубина деревьев
    random_state=42,
    n_jobs=-1           # Используем все ядра процессора
)

model.fit(X_train, y_train)
print("✓ Модель Random Forest успешно обучена!")

# 6. Делаем прогнозы на тестовых данных
y_pred = model.predict(X_test)

# 7. Оцениваем качество модели
print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ ПРОГНОЗИРОВАНИЯ СПРОСА")
print("=" * 60)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Средняя абсолютная ошибка (MAE): {mae:.2f} порций")
print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"Корень из MSE (RMSE): {rmse:.2f} порций")
print(f"Коэффициент детерминации R²: {r2:.4f}")
print("\nИнтерпретация:")
print("- MAE/RMSE: средняя ошибка прогноза в порциях")
print("- R²: близко к 1.0 = отличная модель, близко к 0 = плохая")
print(f"- Модель ошибается в среднем на ±{rmse:.1f} порций")

# 8. Сохраняем модель в файл для будущего использования
with open('demand_forecast_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✓ Модель сохранена в файл 'demand_forecast_model.pkl'")

# 9. Визуализируем важность признаков
print("\n" + "=" * 60)
print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ДЛЯ ПРОГНОЗА")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-5 самых важных признаков для прогноза спроса:")
for i, row in feature_importance.head().iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}")

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['feature'][:10], 
                feature_importance['importance'][:10], 
                color='lightgreen')
plt.xlabel('Важность признака')
plt.title('Топ-10 важнейших признаков для прогноза спроса', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# 10. Визуализация прогнозов vs факт
print("\n" + "=" * 60)
print("ВИЗУАЛИЗАЦИЯ ПРОГНОЗОВ (первые 50 записей теста)")
print("=" * 60)

plt.figure(figsize=(14, 6))
plt.plot(y_test.values[:50], 'o-', label='Фактический спрос', linewidth=2, markersize=6)
plt.plot(y_pred[:50], 's-', label='Прогноз модели', linewidth=2, markersize=6, alpha=0.8)
plt.fill_between(range(50), y_pred[:50]-rmse, y_pred[:50]+rmse, alpha=0.2, color='gray', 
                 label=f'Доверительный интервал (±{rmse:.1f})')
plt.xlabel('Номер тестового примера')
plt.ylabel('Спрос (порции)')
plt.title('Сравнение фактического спроса и прогноза модели', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('demand_forecast_vs_actual.png', dpi=150)
plt.show()

print("\n✅ МОДЕЛЬ ПРОГНОЗИРОВАНИЯ СПРОСА ГОТОВА К ИСПОЛЬЗОВАНИЮ!")