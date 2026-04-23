# forteBank_ml_tastks

# DR Macro Model

Модель прогнозирования вероятности дефолта на основе макрофакторов.

## Структура проекта
project/
│
├── data/                       # сырые датасеты, обработанные датасеты после EDA 
├── mlflow/                     # артефакты mlflow, логи mlflow
├── notebooks/                  # только EDA и анализ, эксперименты
├── src/
│   ├── data/                   # функции обработки данных
│   ├── features/               # analyze_and_select_features и прочий отбор признаков 
│   ├── models/                 # train_and_log функция, метрики, калибровка
│   │
│   └── config.py               # ROOT_DIR, MLFLOW_TRACKING_URI
│
├── .gitignore
├── README.md
├── requirements.txt
└── run_mlflow_server_locally.sh

## Установка
pip install -r requirements.txt

## Запуск MLflow сервера
Запуск MLflow в терминале (Linux)
sh run_mlflow_server_locally.sh

## Запуск обучения
python src/models/train.py

## Данные
- data/raw/ — сырые датасеты
- data/processed/ — обработанные данные

## Модели
- Lasso
- LogisticRegression
- CatBoost