# main.py - ВЕРСИЯ С ДОБАВЛЕНИЕМ НОВЫХ ДАННЫХ
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

print("=" * 60)
print("ОБНОВЛЕНИЕ БАЗЫ ДАННЫХ СТУДЕНЧЕСКОЙ СТОЛОВОЙ")
print("=" * 60)

# Устанавливаем "семя" для воспроизводимости
np.random.seed(42)
random.seed(42)

# Основные параметры
dishes = [
    {"id": 1, "name": "Borshch", "base_cost": 55, "category": "Hot"},
    {"id": 2, "name": "Pure with cutlet", "base_cost": 70, "category": "Hot"},
    {"id": 3, "name": "Grechka with chicken", "base_cost": 65, "category": "Hot"},
    {"id": 4, "name": "Salad Olivye", "base_cost": 45, "category": "Salad"},
    {"id": 5, "name": "Pizza slice", "base_cost": 80, "category": "Bakery"},
    {"id": 6, "name": "Coffee americano", "base_cost": 40, "category": "Drink"},
    {"id": 7, "name": "Tea", "base_cost": 20, "category": "Drink"},
    {"id": 8, "name": "Bun with poppy", "base_cost": 30, "category": "Bakery"}
]

# 1. Проверяем, существует ли уже файл с данными
filename = "canteen_sales_dataset.csv"

if os.path.exists(filename):
    print("✓ Файл с данными найден. Загружаю...")
    
    # Загружаем существующие данные
    existing_df = pd.read_csv(filename, encoding='cp1251')
    existing_df['date'] = pd.to_datetime(existing_df['date'])
    
    # Находим последнюю дату в данных
    last_date = existing_df['date'].max()
    print(f"  Последняя дата в базе: {last_date.date()}")
    
    # Начинаем со следующего дня после последней даты
    start_date = last_date + timedelta(days=1)
    
else:
    print("✗ Файл с данными не найден. Создаю новый...")
    # Начинаем с 90 дней назад
    start_date = datetime.now() - timedelta(days=90)
    existing_df = pd.DataFrame()

# 2. Определяем конечную дату (вчерашний день)
# Чтобы не генерировать данные на сегодня (он еще не закончился)
end_date = datetime.now() - timedelta(days=1)

# Если начальная дата уже после конечной, значит данные уже актуальны
if start_date > end_date:
    print(f"✓ База данных уже актуальна. Новых данных не требуется.")
    print(f"  Последняя дата: {end_date.date()}")
    print(f"  Количество записей: {len(existing_df)}")
    exit()

print(f"✓ Будет добавлены данные за период:")
print(f"  С: {start_date.date()}")
print(f"  По: {end_date.date()}")
print(f"  Дней: {(end_date - start_date).days + 1}")

# 3. Генерируем новые данные
new_records = []
current_date = start_date
days_to_generate = (end_date - start_date).days + 1

print(f"\nГенерирую новые данные ({days_to_generate} дней)...")

for day_offset in range(days_to_generate):
    current_date_for_gen = start_date + timedelta(days=day_offset)
    day_of_week = current_date_for_gen.weekday()
    
    # Определяем "тип" дня
    is_weekend = day_of_week >= 5
    # Сессия в январе и июне
    is_session = current_date_for_gen.month in [1, 6]
    is_holiday = random.random() < 0.05
    
    # Базовый множитель спроса
    if is_holiday:
        demand_multiplier = 0.3
    elif is_weekend:
        demand_multiplier = 0.4
    elif is_session:
        demand_multiplier = 0.6
    else:
        demand_multiplier = 1.0
    
    # Для каждого блюда
    for dish in dishes:
        # Базовый спрос
        base_demand = random.randint(10, 50)
        adjusted_demand = int(base_demand * demand_multiplier)
        final_demand = max(1, adjusted_demand + random.randint(-5, 5))
        
        # Цена
        if is_weekend or is_holiday:
            markup = random.uniform(1.2, 1.5)
        else:
            markup = random.uniform(1.5, 2.2)
            
        selling_price = round(dish["base_cost"] * markup, 2)
        revenue = selling_price * final_demand
        profit = (selling_price - dish["base_cost"]) * final_demand
        
        # Создаем запись
        record = {
            "date": current_date_for_gen.date(),
            "day_of_week": day_of_week,
            "dish_id": dish["id"],
            "dish_name": dish["name"],
            "category": dish["category"],
            "base_cost": dish["base_cost"],
            "selling_price": selling_price,
            "demand": final_demand,
            "revenue": revenue,
            "profit": profit,
            "is_weekend": is_weekend,
            "is_session": is_session,
            "is_holiday": is_holiday,
            "temperature": random.randint(-15, 30)
        }
        new_records.append(record)

# 4. Создаем DataFrame с новыми данными
new_df = pd.DataFrame(new_records)
new_df['date'] = pd.to_datetime(new_df['date'])

print(f"✓ Сгенерировано {len(new_df)} новых записей")

# 5. Объединяем старые и новые данные
if not existing_df.empty:
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"✓ Объединено с существующими {len(existing_df)} записями")
else:
    combined_df = new_df
    print("✓ Создана новая база данных")

# 6. Сохраняем объединенные данные
combined_df.to_csv(filename, index=False, encoding='cp1251')
print(f"✓ База данных сохранена. Всего записей: {len(combined_df)}")

# 7. Статистика
print(f"\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ:")
print(f"  Общий период: с {combined_df['date'].min().date()} по {combined_df['date'].max().date()}")
print(f"  Всего дней: {combined_df['date'].dt.date.nunique()}")
print(f"  Всего записей: {len(combined_df)}")
print(f"  Уникальных блюд: {combined_df['dish_id'].nunique()}")
print(f"  Средний спрос: {combined_df['demand'].mean():.1f} порций/день")
print(f"  Средняя прибыль: {combined_df['profit'].mean():.1f} руб./запись")

# 8. Простая визуализация
plt.figure(figsize=(12, 5))

# График общего количества записей по дням
plt.subplot(1, 2, 1)
daily_counts = combined_df.groupby(combined_df['date'].dt.date).size()
plt.plot(daily_counts.index, daily_counts.values, 'b-', alpha=0.7)
plt.xlabel('Дата')
plt.ylabel('Количество записей')
plt.title('Количество записей по дням')
plt.xticks(rotation=45)

# График средней прибыли по дням
plt.subplot(1, 2, 2)
daily_profit = combined_df.groupby(combined_df['date'].dt.date)['profit'].sum()
plt.plot(daily_profit.index, daily_profit.values, 'g-', alpha=0.7)
plt.xlabel('Дата')
plt.ylabel('Прибыль, руб.')
plt.title('Общая прибыль по дням')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('database_stats.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ БАЗА ДАННЫХ УСПЕШНО ОБНОВЛЕНА!")
print(f"   Файл: {filename}")
print(f"   График: database_stats.png")