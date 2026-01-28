# price_optimization.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("МОДЕЛЬ ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ")
print("=" * 60)

# 1. Загружаем данные
df = pd.read_csv('canteen_sales_dataset.csv')
print(f"✓ Загружено {len(df)} записей о продажах")

# 2. Анализируем эластичность спроса по цене для каждого блюда
# Эластичность = на сколько % изменится спрос при изменении цены на 1%
print("\n" + "-" * 60)
print("АНАЛИЗ ЭЛАСТИЧНОСТИ СПРОСА ПО ЦЕНЕ")
print("-" * 60)

dishes_analysis = []
for dish_id in df['dish_id'].unique():
    dish_data = df[df['dish_id'] == dish_id]
    
    if len(dish_data) > 10:  # Анализируем только блюда с достаточным количеством данных
        # Простая линейная регрессия: спрос ~ цена
        X = dish_data[['selling_price']].values
        y = dish_data['demand'].values
        
        # Нормализуем данные
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Обучаем модель
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Коэффициент регрессии показывает эластичность
        # Отрицательный коэффициент = при росте цены спрос падает
        elasticity = model.coef_[0] * scaler.scale_[0]  # Преобразуем обратно в оригинальный масштаб
        
        # Средние значения для блюда
        avg_price = dish_data['selling_price'].mean()
        avg_demand = dish_data['demand'].mean()
        avg_profit = dish_data['profit'].mean()
        
        dishes_analysis.append({
            'dish_id': dish_id,
            'dish_name': dish_data['dish_name'].iloc[0],
            'category': dish_data['category'].iloc[0],
            'elasticity': elasticity,
            'avg_price': avg_price,
            'avg_demand': avg_demand,
            'avg_profit': avg_profit,
            'data_points': len(dish_data)
        })

# Создаем DataFrame с результатами анализа
elasticity_df = pd.DataFrame(dishes_analysis)
elasticity_df['elasticity_abs'] = abs(elasticity_df['elasticity'])
elasticity_df = elasticity_df.sort_values('elasticity_abs', ascending=False)

print("\nАнализ эластичности по блюдам (от самых чувствительных к цене):")
print(elasticity_df[['dish_name', 'category', 'elasticity', 'avg_price', 'avg_demand']].to_string())

# 3. Функция для расчета оптимальной цены
def calculate_optimal_price(dish_info, predicted_demand_func, cost_price):
    """
    Рассчитывает оптимальную цену для максимизации прибыли.
    
    Параметры:
    - dish_info: информация о блюде (эластичность, средний спрос)
    - predicted_demand_func: функция прогноза спроса в зависимости от цены
    - cost_price: себестоимость блюда
    
    Возвращает оптимальную цену и ожидаемую прибыль.
    """
    
    # Функция прибыли: (цена - себестоимость) * спрос(цена)
    def profit_function(price):
        predicted_demand = predicted_demand_func(price)
        return -(price - cost_price) * predicted_demand  # Минус для минимизации
    
    # Ищем оптимальную цену в разумных пределах
    price_bounds = (cost_price * 1.1, cost_price * 3.0)  # От +10% до +200% от себестоимости
    
    result = optimize.minimize_scalar(
        profit_function,
        bounds=price_bounds,
        method='bounded'
    )
    
    optimal_price = result.x
    max_profit = -result.fun  # Убираем минус
    
    return optimal_price, max_profit

# 4. Пример: оптимизация цены для конкретного блюда
print("\n" + "-" * 60)
print("ПРИМЕР РАСЧЕТА ОПТИМАЛЬНОЙ ЦЕНЫ")
print("-" * 60)

# Выбираем блюдо для примера (например, борщ)
sample_dish = elasticity_df.iloc[0]
dish_id = sample_dish['dish_id']
dish_name = sample_dish['dish_name']

# Данные по этому блюду
dish_data = df[df['dish_id'] == dish_id]
cost_price = dish_data['base_cost'].iloc[0]
current_avg_price = sample_dish['avg_price']
current_avg_profit = sample_dish['avg_profit']

print(f"Анализируем блюдо: {dish_name}")
print(f"  • Себестоимость: {cost_price:.2f} руб.")
print(f"  • Текущая средняя цена: {current_avg_price:.2f} руб.")
print(f"  • Текущая средняя прибыль: {current_avg_profit:.2f} руб.")
print(f"  • Эластичность: {sample_dish['elasticity']:.4f}")

# Создаем простую модель спроса от цены для этого блюда
X_price = dish_data[['selling_price']].values.reshape(-1, 1)
y_demand = dish_data['demand'].values

price_model = LinearRegression()
price_model.fit(X_price, y_demand)

# Функция прогноза спроса в зависимости от цены
def demand_prediction(price):
    return max(1, price_model.predict([[price]])[0])

# Рассчитываем оптимальную цену
optimal_price, expected_profit = calculate_optimal_price(
    sample_dish, demand_prediction, cost_price
)

print(f"\nРЕЗУЛЬТАТ ОПТИМИЗАЦИИ:")
print(f"  • Рекомендуемая цена: {optimal_price:.2f} руб.")
print(f"  • Ожидаемая прибыль: {expected_profit:.2f} руб.")
print(f"  • Изменение цены: {((optimal_price - current_avg_price)/current_avg_price*100):+.1f}%")
print(f"  • Рост прибыли: {((expected_profit - current_avg_profit)/current_avg_profit*100):+.1f}%")

# 5. Визуализация: как цена влияет на спрос и прибыль
print("\n" + "-" * 60)
print("ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТИ СПРОСА И ПРИБЫЛИ ОТ ЦЕНЫ")
print("-" * 60)

# Генерируем диапазон цен
price_range = np.linspace(cost_price * 1.05, cost_price * 2.5, 50)
demand_range = [demand_prediction(p) for p in price_range]
profit_range = [(p - cost_price) * d for p, d in zip(price_range, demand_range)]

plt.figure(figsize=(12, 5))

# График 1: Цена vs Спрос
plt.subplot(1, 2, 1)
plt.plot(price_range, demand_range, 'b-', linewidth=2, label='Прогноз спроса')
plt.axvline(x=current_avg_price, color='r', linestyle='--', alpha=0.7, label='Текущая средняя цена')
plt.axvline(x=optimal_price, color='g', linestyle='--', alpha=0.7, label='Оптимальная цена')
plt.xlabel('Цена, руб.')
plt.ylabel('Спрос, порции')
plt.title(f'Эластичность спроса: {dish_name}', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Цена vs Прибыль
plt.subplot(1, 2, 2)
plt.plot(price_range, profit_range, 'g-', linewidth=2, label='Прибыль')
plt.axvline(x=current_avg_price, color='r', linestyle='--', alpha=0.7, label='Текущая цена')
plt.axvline(x=optimal_price, color='g', linestyle='--', alpha=0.7, label='Оптимальная цена')
plt.axhline(y=expected_profit, color='g', linestyle=':', alpha=0.5, label='Макс. прибыль')
plt.xlabel('Цена, руб.')
plt.ylabel('Прибыль, руб.')
plt.title('Зависимость прибыли от цены', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_optimization_example.png', dpi=150)
plt.show()

# 6. Рекомендации по ценообразованию для всех блюд
print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИИ ПО ДИНАМИЧЕСКОМУ ЦЕНООБРАЗОВАНИЮ")
print("=" * 60)

recommendations = []
for _, dish_row in elasticity_df.iterrows():
    dish_id = dish_row['dish_id']
    dish_data = df[df['dish_id'] == dish_id]
    
    if len(dish_data) > 10:
        cost_price = dish_data['base_cost'].iloc[0]
        current_price = dish_row['avg_price']
        
        # Простая оптимизация для каждого блюда
        X_price = dish_data[['selling_price']].values.reshape(-1, 1)
        y_demand = dish_data['demand'].values
        
        price_model = LinearRegression()
        price_model.fit(X_price, y_demand)
        
        def dish_demand_prediction(price):
            return max(1, price_model.predict([[price]])[0])
        
        opt_price, opt_profit = calculate_optimal_price(
            dish_row, dish_demand_prediction, cost_price
        )
        
        price_change_pct = (opt_price - current_price) / current_price * 100
        
        # Определяем рекомендацию
        if price_change_pct > 5:
            action = "ПОВЫСИТЬ цену"
            reason = "Низкая эластичность, можно увеличить маржу"
        elif price_change_pct < -5:
            action = "ПОНИЗИТЬ цену"
            reason = "Высокая эластичность, рост спроса компенсирует снижение цены"
        else:
            action = "ОСТАВИТЬ цену"
            reason = "Цена близка к оптимальной"
        
        recommendations.append({
            'Блюдо': dish_row['dish_name'],
            'Категория': dish_row['category'],
            'Себест.': f"{cost_price:.1f} руб.",
            'Тек. цена': f"{current_price:.1f} руб.",
            'Опт. цена': f"{opt_price:.1f} руб.",
            'Изменение': f"{price_change_pct:+.1f}%",
            'Действие': action,
            'Обоснование': reason
        })

# Создаем и выводим таблицу рекомендаций
rec_df = pd.DataFrame(recommendations)
print("\nРекомендации по ценообразованию на основе анализа данных:")
print(rec_df.to_string(index=False))

# Сохраняем рекомендации в файл
rec_df.to_csv('price_recommendations.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ Рекомендации сохранены в 'price_recommendations.csv'")

print("\n" + "=" * 60)
print("КРАТКИЙ АЛГОРИТМ ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ")
print("=" * 60)
print("""
1. ЕЖЕДНЕВНО:
   - Получить прогноз спроса на завтра (из predict_demand.py)
   - Учесть факторы: день недели, сессия, погода

2. ДЛЯ КАЖДОГО БЛЮДА:
   - Рассчитать оптимальную цену по формуле:
       Оптимальная цена = argmax[(цена - себестоимость) * прогнозируемый_спрос(цена)]
   
3. КОРРЕКТИРОВКИ:
   - Для блюд с низкой эластичностью (студенты все равно купят): повышать цену
   - Для блюд с высокой эластичностью: снижать цену для увеличения спроса
   - Учитывать остатки: снижать цену на блюда с истекающим сроком годности

4. МОНИТОРИНГ:
   - Сравнивать фактические продажи с прогнозом
   - Обновлять модель раз в месяц на новых данных
""")

print("\n✅ СИСТЕМА ДИНАМИЧЕСКОГО ЦЕНООБРАЗОВАНИЯ ГОТОВА!")


