# forteBank_ml_tastks

# DR Macro Model

Модель прогнозирования вероятности дефолта на основе макрофакторов.

## Структура проекта
project/
│
├── data/                       # сырые датасеты, обработанные датасеты после EDA 
├── mlflow/                     # артефакты mlflow, логи mlflow
├── notebooks/                  
|   ├── EDA default_probability # Тут только EDA 
|   └── EDA scoring_model       # Тут EDA и обучение с логированием метрик
├── src/
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

## Запуск обучения для default_probability
python src/models/train_default_probability.py



## Данные
- data/raw/ — сырые датасеты
- data/processed/ — обработанные данные

## Модели
- Lasso
- LogisticRegression
- CatBoost