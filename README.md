# Прогнозирование свойств бетонной смеси

Репозиторий для предсказания прочности бетона по составу смеси.
Метрики final_solution.ipynb
## Какая модель

- Генератор: **AutoregressiveGenerator** (дни 1, 3, 7, 28)
- Дискриминатор: **NEAT + BNN**
- Гарантия монотонности по времени: $\hat{y}_k = \hat{y}_{k-1} + \text{softplus}(\delta_k)$
- Основной таргет: день 28 (повышенный вес в loss)

## Что сейчас использовать

- Модель: **v18_final** на полном датасете (1808 строк)
- Конфиг: `examples/gan_v18_final.json`


## Быстрый запуск

```bash
# 1) Установка зависимостей
pip install -r requirements.txt

```

## Обучение модели

```bash
python main.py train_gan --config examples/gan_v18_final.json --gan-dir artifacts/train_gan_v18_final
```

## Логи и контроль прогресса

```bash
tail -f artifacts/train_gan_v18_final/training.log

tail -n 80 artifacts/train_gan_v18_final/training.log

grep -n "Running generation\|HEARTBEAT\|FINISH\|EXIT_CODE" artifacts/train_gan_v18_final/training.log | tail -n 40
```


## Где результаты после обучения
 
В папке `artifacts/train_gan_v18_final/`:

- `training_summary.json` — итоговые метрики
- `validation_predictions.csv` — предсказания на валидации
- `generator.pt` — веса генератора

## Ключевые файлы

- `final_solution.ipynb` — финальный ноутбук с архитектурой и метриками
- `examples/gan_v18_final.json` — основной финальный конфиг обучения
- `data/processed/all_sources_strength_v16_full.csv` — полный датасет (1808 строк)
- `materialgen/train_gan.py` — реализация обучения GAN

