# MLOps: учебный проект команды 2

Учебный ML-проект курса «Управление процессами машинного обучения (MLOps)».
Тема проекта: Система рекомендаций
Описание: MLOps система для рекомендации товаров/контента пользователям

## Требования

- Git
- Python 3.12 (версия зафиксирована в `.python-version`)
- Docker пока не требуется

## Установка

Клонирование:

```bash
git clone https://github.com/kucheranton200/T26_MLOps.git
cd T26_MLOps
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux или macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Проверки

```bash
python -m ruff check .
python -m pytest -q
```

## Запуск

```bash
python -m mlops_project
```

Ожидаемый вывод: `mlops project is ready`.

## Данные и проверка качества

Происхождение, состав и ограничения набора: [docs/data-card.md](docs/data-card.md).
Машинный контракт данных: [data/contract.json](data/contract.json).

Проверка данных (код завершения 0 только для корректного набора):

```bash
python -m mlops_project.validate_data --data data/<файл>.csv --contract data/contract.json --report reports/validation/lr2.json
```

В PowerShell команду можно записать в одну строку, как выше. Отчёт содержит версию контракта,
SHA-256 проверенного файла, число строк и результаты отдельных проверок.

Тесты контракта и линтер:

```bash
python -m pytest -q tests/test_data_contract.py
python -m ruff check src tests
```

## Версионирование контракта

| Изменение | Версия |
|---|---|
| Исправление описания без изменения проверки | patch |
| Совместимое добавление необязательного поля | minor |
| Удаление или переименование поля, смена типа или смысла | major |

## Отчёты валидации

Отчёты `reports/validation/*.json` хранятся в Git. Отчёт создаётся командой выше и также сохраняется как артефакт CI. Воспроизвести его по tag `lr2-v1` можно запуском этой же команды.

## Какие данные нельзя добавлять в Git

Полные закрытые или лицензионно ограниченные наборы, персональные данные, секреты и токены,
выгрузки с неизвестным происхождением, воспроизводимые обработанные копии (`data/raw/`,
`data/processed/`). В репозитории хранится только разрешённый обезличенный образец.

## Структура

```
.github/        шаблоны issue и PR, CI workflow
data/           контракт данных и разрешённый образец (data/raw и data/processed не в Git)
docs/           ADR, карточка данных, evidence cards, роли команды
reports/        отчёты валидации
src/            исходный код пакета mlops_project
tests/          тесты
```

## Как работать с репозиторием

Правила веток, commit, pull request и review: [CONTRIBUTING.md](CONTRIBUTING.md).
