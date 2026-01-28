# run_all.py - запускает все скрипты по порядку
import subprocess
import sys

scripts = ['main.py', 'predict_demand.py', 'price_optimization.py', 'daily_recommendations.py']

for script in scripts:
    print(f"\n{'='*60}")
    print(f"ЗАПУСК: {script}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, 
                              text=True,
                              encoding='utf-8')
        
        if result.returncode == 0:
            print(f"✅ {script} выполнен успешно")
            # Выводим последние 5 строк вывода
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:
                    if line.strip():
                        print(f"   {line}")
        else:
            print(f"❌ Ошибка в {script}:")
            print(result.stderr[:500])  # Первые 500 символов ошибки
            break
            
    except FileNotFoundError:
        print(f"❌ Файл {script} не найден")
        break
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        break

print(f"\n{'='*60}")
print("ВСЕ СКРИПТЫ ВЫПОЛНЕНЫ!")
print(f"{'='*60}")