# retrain_model.py - переобучение модели на новых данных
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print(f"Переобучение модели на новых данных - {datetime.now()}")
print("="*60)

# 1. Загружаем ВСЕ данные
try:
    df = pd.read_csv('canteen_sales_dataset.csv', parse_dates=['date'], encoding='cp1251')
    print(f"✅ Загружено записей: {len(df)}")
    print(f"📅 Диапазон дат: {df['date'].min().date()} - {df['date'].max().date()}")
except:
    print("❌ Ошибка: файл 'canteen_sales_dataset.csv' не найден")
    print("   Сначала запустите main.py")
    exit()

# 2. Подготовка данных
df['month'] = df['date'].dt.month
df['day_of_month'] = df['date'].dt.day
df['prev_day_demand'] = df.groupby('dish_id')['demand'].shift(1)
df['prev_day_demand'].fillna(df['demand'].mean(), inplace=True)

# 3. Берем последние 90% данных (самые свежие)
train_size = int(len(df) * 0.9)
recent_df = df.tail(train_size)
print(f"📊 Использую {len(recent_df)} последних записей для обучения")

# 4. Подготовка признаков
feature_columns = [
    'day_of_week', 'month', 'day_of_month', 'is_weekend',
    'is_session', 'temperature', 'base_cost', 'selling_price',
    'prev_day_demand', 'dish_id'
]

X = recent_df[feature_columns]
y = recent_df['demand']

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"📈 Разделение данных: обучающих - {X_train.shape[0]}, тестовых - {X_test.shape[0]}")

# 6. Обучение новой модели
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Модель обучена на свежих данных")

# 7. Оценка модели
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Точность модели: MAE = {mae:.2f} порций")

# 8. Сохранение модели
model_filename = f"demand_model_retrained_{datetime.now().strftime('%Y%m%d')}.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

# Также обновляем основной файл модели
with open('demand_forecast_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"💾 Модель сохранена в файлы:")
print(f"   - {model_filename}")
print(f"   - demand_forecast_model.pkl (основной)")

# 9. Сравнение с предыдущей моделью (если есть)
try:
    with open('demand_forecast_model.pkl', 'rb') as f:
        old_model = pickle.load(f)
    
    # Простое сравнение
    print("\n" + "="*60)
    print("СРАВНЕНИЕ С ПРЕДЫДУЩЕЙ МОДЕЛЬЮ:")
    print(f"Новая модель: {len(recent_df)} последних записей")
    print(f"Старая модель: обучена на {len(df)} записях")
    print("✅ Модель обновлена на актуальных данных")
    
except:
    print("ℹ️ Предыдущая модель не найдена, создана новая")

print("\n" + "="*60)
print("ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("="*60)