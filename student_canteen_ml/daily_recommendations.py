# daily_recommendations.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("СИСТЕМА РЕКОМЕНДАЦИЙ ДЛЯ СТУДЕНЧЕСКОЙ СТОЛОВОЙ")
print(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
print("=" * 70)

# 1. Загружаем обученную модель прогнозирования спроса
print("\n📊 ЗАГРУЗКА МОДЕЛЕЙ...")
try:
    with open('demand_forecast_model.pkl', 'rb') as f:
        demand_model = pickle.load(f)
    print("✓ Модель прогнозирования спроса загружена")
except:
    print("✗ Файл 'demand_forecast_model.pkl' не найден")
    print("  Запустите сначала predict_demand.py")
    exit()

# 2. Загружаем исторические данные
df = pd.read_csv('canteen_sales_dataset.csv', parse_dates=['date'])
print("✓ Исторические данные загружены")

# 3. Подготавливаем данные для прогноза на завтра
tomorrow = datetime.now() + timedelta(days=1)
print(f"\n📅 ПРОГНОЗ НА: {tomorrow.strftime('%d.%m.%Y (%A)')}")

# Определяем параметры завтрашнего дня
day_of_week = tomorrow.weekday()
is_weekend = 1 if day_of_week >= 5 else 0
is_session = 1 if tomorrow.month in [1, 6, 12] else 0  # Предполагаемая сессия
is_holiday = 0  # Можно добавить календарь праздников
temperature = 15  # Прогноз погоды - можно получать из API

# 4. Создаем DataFrame для прогноза по всем блюдам
forecast_data = []
dishes_info = df[['dish_id', 'dish_name', 'category', 'base_cost']].drop_duplicates()

for _, dish in dishes_info.iterrows():
    # Берем последние известные значения для этого блюда
    dish_history = df[df['dish_id'] == dish['dish_id']].iloc[-1]
    
    # Формируем признаки для модели
    features = {
        'day_of_week': day_of_week,
        'month': tomorrow.month,
        'day_of_month': tomorrow.day,
        'is_month_start': 1 if tomorrow.day == 1 else 0,
        'is_month_end': 1 if tomorrow.day in [28, 29, 30, 31] else 0,
        'is_weekend': is_weekend,
        'is_session': is_session,
        'is_holiday': is_holiday,
        'temperature': temperature,
        'base_cost': dish['base_cost'],
        'selling_price': dish_history['selling_price'],  # Текущая цена
        'prev_day_demand': dish_history['demand'],  # Спрос сегодня
        'dish_id': dish['dish_id']
    }
    
    forecast_data.append(features)

forecast_df = pd.DataFrame(forecast_data)

# 5. Делаем прогноз спроса
demand_predictions = demand_model.predict(forecast_df)

# 6. Рассчитываем рекомендации по цене
print("\n💰 РЕКОМЕНДАЦИИ ПО ЦЕНООБРАЗОВАНИЮ:")
print("-" * 70)

recommendations = []
for i, (_, dish) in enumerate(dishes_info.iterrows()):
    predicted_demand = max(1, round(demand_predictions[i]))
    current_price = forecast_df.iloc[i]['selling_price']
    cost_price = dish['base_cost']
    
    # Простая логика динамического ценообразования
    if predicted_demand > 30:  # Высокий спрос
        recommended_price = current_price * 1.1  # +10%
        reason = "Высокий прогнозируемый спрос"
    elif predicted_demand < 10:  # Низкий спрос
        recommended_price = current_price * 0.9  # -10%
        reason = "Низкий прогнозируемый спрос (стимулирование продаж)"
    else:
        recommended_price = current_price
        reason = "Спрос в норме"
    
    # Ограничиваем минимальную наценку
    min_price = cost_price * 1.2  # Минимум +20% к себестоимости
    recommended_price = max(min_price, recommended_price)
    
    # Рассчитываем ожидаемую прибыль
    expected_profit = (recommended_price - cost_price) * predicted_demand
    
    recommendations.append({
        'Блюдо': dish['dish_name'],
        'Категория': dish['category'],
        'Себест.': f"{cost_price:.1f}",
        'Тек. цена': f"{current_price:.1f}",
        'Рек. цена': f"{recommended_price:.1f}",
        'Прогноз спроса': predicted_demand,
        'Ожид. прибыль': f"{expected_profit:.1f}",
        'Логика': reason
    })

# 7. Выводим рекомендации
rec_df = pd.DataFrame(recommendations)

# Сортируем по ожидаемой прибыли
rec_df['Ожид. прибыль_num'] = rec_df['Ожид. прибыль'].str.replace('руб', '').astype(float)
rec_df = rec_df.sort_values('Ожид. прибыль_num', ascending=False)
rec_df = rec_df.drop('Ожид. прибыль_num', axis=1)

print(rec_df.to_string(index=False))

# 8. Сводная статистика
print("\n" + "=" * 70)
print("📈 СВОДНАЯ СТАТИСТИКА НА ЗАВТРА")
print("=" * 70)

total_expected_demand = rec_df['Прогноз спроса'].sum()
total_expected_profit = rec_df['Ожид. прибыль'].str.replace('руб', '').astype(float).sum()
avg_price_change = ((pd.to_numeric(rec_df['Рек. цена']) - 
                     pd.to_numeric(rec_df['Тек. цена'])) / 
                    pd.to_numeric(rec_df['Тек. цена']) * 100).mean()

print(f"Общий прогнозируемый спрос: {total_expected_demand:.0f} порций")
print(f"Ожидаемая общая прибыль: {total_expected_profit:.1f} руб.")
print(f"Среднее изменение цены: {avg_price_change:+.1f}%")
print(f"Количество блюд в меню: {len(rec_df)}")

# 9. Специальные рекомендации
print("\n" + "=" * 70)
print("🎯 СПЕЦИАЛЬНЫЕ РЕКОМЕНДАЦИИ")
print("=" * 70)

# Топ-3 самых прибыльных блюда
top_profit = rec_df.head(3)
print("\n✅ ТОП-3 САМЫХ ПРИБЫЛЬНЫХ БЛЮДА (делать упор на продажах):")
for _, row in top_profit.iterrows():
    print(f"  • {row['Блюдо']}: {row['Прогноз спроса']} порций, {row['Ожид. прибыль']} руб.")

# Блюда с низким спросом
low_demand = rec_df[rec_df['Прогноз спроса'] < 10]
if len(low_demand) > 0:
    print("\n⚠️  ВНИМАНИЕ: БЛЮДА С НИЗКИМ ПРОГНОЗИРУЕМЫМ СПРОСОМ (<10 порций):")
    for _, row in low_demand.iterrows():
        print(f"  • {row['Блюдо']}: {row['Прогноз спроса']} порций - рассмотрите акции")

# 10. Сохраняем отчет
report_filename = f"recommendations_{tomorrow.strftime('%Y%m%d')}.csv"
rec_df.to_csv(report_filename, index=False, encoding='utf-8-sig')

print(f"\n💾 Отчет сохранен в файл: {report_filename}")

print("\n" + "=" * 70)
print("✅ СИСТЕМА ГОТОВА! РЕКОМЕНДАЦИИ СФОРМИРОВАНЫ.")
print("=" * 70)