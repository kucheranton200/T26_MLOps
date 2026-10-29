# Evidence Card ЛР3 — Кучер Антон (@kucheranton200)

**Роль в ЛР3:** Envoriment engineer 
**Команда:** team-2  
**Репозиторий:** https://github.com/kucheranton200/T26_MLOps  
**Ветки:** `docs/lr3-dvc-readme`  
**Issues:** [#55](https://github.com/kucheranton200/T26_MLOps/issues/55)  
**Pull Requests:** [#56](https://github.com/kucheranton200/T26_MLOps/issues/56)  
**Тег:** `lr3-v1` _(ставится координатором @GlooshSl после слияния всех PR)_

---

## 1. Что сделано

| Артефакт / Задача | Путь | Назначение |
|---|---|---|
| Readme | `README.md` | Добавлена инструкция для воспроизведения |
| Лок файл | `dvc.lock` | Обновлен для текушей конфигурации |
| Личная evidence-card | `docs/evidence/lr3/lr3-evidence-card-kerty0.md` | Подтверждение выполнения роли Envoriment engineer, команд, решений и анализа рисков |

---

## 2. Команды запуска и проверки

```bash
# Установка зависимостей
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Воспроизведение конвейера DVC
dvc repro

# Проверка статуса стадий DVC
dvc status

# Проверка корректности правил игнорирования Git
git check-ignore -v data/processed/flights_processed.csv data/split/X_train.csv models/model.pkl models/scaler.pkl
git check-ignore -v data/flights_sample.csv data/contract.json params.yaml dvc.yaml dvc.lock

# Проверка качества кода и запуск тестов
ruff check .
pytest -q
```

---

## 3. Метрики baseline (`reports/metrics.json`)

Параметры: `random_state=42`, `test_size=0.2`, `model_type=logistic_regression`, `C=1.0`

| Метрика | Значение |
|---|---|
| ROC AUC | **0.666667** |
| F1 | **0.5** |
| Brier Score | **0.228108** |
| Accuracy | **0.6** |
| n_test | 5 |
| threshold | 0.5 |

> **Примечание:** Датасет `flights_sample.csv` содержит 24 строки (синтетические данные ЛР2).  
> Маленький размер тестовой выборки (5 объектов) ожидаемо даёт нестабильные метрики.  
> Производительность модели показывает, что данных недостаточно для надёжного обучения,  
> но функциональность конвейера подтверждена.

---

## 4. Подтверждение идемпотентности (повторный `dvc repro`)

```
$ dvc repro

Stage 'validate' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Stage 'split' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

DVC пропускает все стадии, так как хеши зависимостей, параметров и выходных файлов  
не изменились. Это подтверждает корректную работу механизма отслеживания состояния.

---

## 5. Структура конвейера

```
validate ──→ preprocess ──→ split ──→ train ──→ evaluate
    │              │           │         │          │
 lr3.json   processed.csv  X_train  model.pkl  metrics.json
                          X_test
                          y_train
                          y_test
                          scaler.pkl
``` Исходные файлы, параметры, пайплайн и lock-файл НЕ игнорируются (вывод пуст):
$ git check-ignore -v data/flights_sample.csv data/contract.json params.yaml dvc.yaml dvc.lock
(пустой вывод)
```

В Git коммитятся только управляющие файлы конвейера (`dvc.yaml`, `dvc.lock`, `params.yaml`) и отчёты с `cache: false` (`reports/metrics.json`, `reports/validation/lr3.json`).

---

## 6. Тесты

```
$ pytest
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.7.0
rootdir: /home/kerty/Projects/T26_MLOps
configfile: pyproject.toml
testpaths: tests
plugins: platformdirs-4.13.7, hydra-core-1.3.8
collected 29 items

tests/test_data_contract.py
tests/test_pipeline.py
tests/test_smoke.py

29 passed in 1.17s
```

---

## 7. Итоги и чек-лист готовности

- [x] Иструкция вопроизведения изменена и проверена (Issue #55, PR #56)
- [x] Зафиксировано архитектурное решение по данным без внешнего storage
- [x] CI в GitHub Actions зелёный для всех триггеров
- [x] `ruff check .` и `pytest -q` проходят успешно
- [x] Создана персональная evidence-card

---

_Evidence card заполнена: 2026-10-26. Контрольный тег `lr3-v1` и итоговый SHA будут проставлены координатором @GlooshSl после завершения этапа слияния всех PR ЛР3._
