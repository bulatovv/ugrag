# Retrieval Experiments

Структура репозитория
------------

    ├── README.md                   <- описание проекта
    │
    ├── ugrag                       <- наработки по загрузке данных
    │
    ├── experiments.ipynb                  <- основной ноутбук с экспериментами
    │
    ├── docker-compose.yml          <- docker конфигурация для поднятия векторной БД и elasticsearch
    │
    ├── requirements.txt            <- system requirements for main.ipynb
    │
    └── pyproject.toml              <- system requirements for ugrag

--------

## Как запустить проект 

1. Клонируйте репозиторий:

Сначала клонируйте репозиторий на ваш компьютер с помощью команды:

    git clone <repository_url>
    cd <repository_directory>
 
2. Подготовка проекта:

Чтобы подготовить проект к экспериментам, выполните следующую команду:

    docker compose up db 
    docker compose up elasticsearch

В случае, если ваша версия Docker ниже, попробуйте запустить через docker-compose

3. Эксперименты

Загрузка данных и проведение экспериментов содержится в файле main.ipynb 
##### TODO разнести все по скриптам
