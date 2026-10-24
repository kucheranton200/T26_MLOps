# Evidence Card ЛР3 — Кныш Радион (@Z0rko32)

**Роль в ЛР3:** Quality engineer  
**Команда:** team-2  
**Репозиторий:** https://github.com/kucheranton200/T26_MLOps  
**Ветка:** `lab3/dvc-pipeline-zorko32`  
**Базовый коммит main:** `db4d347645768b7ac948bcee4b6cf234d48bbb6e`  
**Issue:** [#51](https://github.com/kucheranton200/T26_MLOps/issues/51)  
**PR:** _(будет добавлен после создания)_  
**Тег:** `lr3-v1` _(ставится координатором @GlooshSl после слияния всех PR)_

---

## 1. Что сделано

| Артефакт | Путь | Назначение |
|---|---|---|
| Параметры конвейера | `params.yaml` | Все гиперпараметры, пути, seed; разделы `prepare`, `split`, `train`, `evaluate`, `validate` |
| Граф конвейера DVC | `dvc.yaml` | 5 стадий: `validate` → `preprocess` → `split` → `train` → `evaluate` |
| Предобработка | `src/mlops_project/preprocess.py` | Создание бинарной метки `label = (ArrDelay > 15)`, исключение ArrDelay из признаков |
| Разделение | `src/mlops_project/split_data.py` | Стратифицированный split (20% test, seed=42), StandardScaler только на train |
| Обучение | `src/mlops_project/train.py` | LogisticRegression (или RandomForest) по params.yaml, сохранение `models/model.pkl` |
| Оценка | `src/mlops_project/evaluate.py` | ROC AUC, F1, Brier score, Accuracy → `reports/metrics.json` (cache: false) |
| Тесты | `tests/test_pipeline.py` | 18 тестов: anti-leakage, детерминизм, метрики, структура выходов |
| Нормализация EOL | `.gitattributes` | `* text=auto eol=lf` — предотвращает CRLF-артефакты в хешах DVC |

---

## 2. Команды запуска

```powershell
# Активация окружения (Windows)
.\.venv\Scripts\Activate.ps1

# Воспроизведение конвейера (первый запуск)
python -m dvc repro

# Просмотр метрик
python -m dvc metrics show

# Сравнение с предыдущим запуском
python -m dvc metrics diff
python -m dvc params diff

# Тесты
pytest -q

# Линтинг
ruff check src/ tests/
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
$ python -m dvc repro

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

## 5. Контрольный эксперимент (`random_state: 42 → 99`)

Изменение `random_state` в секции `split` и `train` в `params.yaml`:

```
# В params.yaml: random_state: 99 вместо 42
$ python -m dvc repro

Stage 'validate' didn't change, skipping
Stage 'preprocess' didn't change, skipping
Running stage 'split':  ...
Running stage 'train':  ...
Running stage 'evaluate': ...
```

**Вывод:** DVC пропустил `validate` и `preprocess` (их зависимости не изменились)  
и перезапустил только `split`, `train`, `evaluate` — строго по графу зависимостей.

### `dvc params diff` (random_state=99 vs HEAD)

```
Path         Param                     HEAD    workspace
params.yaml  split.random_state        -       99
params.yaml  train.random_state        -       99
... (новые параметры — т.к. params.yaml был новым файлом)
```

### `dvc metrics diff` (при random_state=99)

```
Path                  Metric       HEAD    workspace    Change
reports/metrics.json  accuracy     -       1.0          -
reports/metrics.json  f1           -       1.0          -
reports/metrics.json  roc_auc      -       1.0          -
reports/metrics.json  brier_score  -       0.21247      -
```

После эксперимента параметры возвращены к baseline (`random_state=42`),  
конвейер восстановил результаты из DVC-кэша без повторного запуска:

```
Stage 'split' is cached - skipping run, checking out outputs
Stage 'train' is cached - skipping run, checking out outputs
Stage 'evaluate' is cached - skipping run, checking out outputs
```

---

## 6. Анализ рисков

### Почему CRLF ломает хеши DVC?

DVC отслеживает зависимости через SHA-256 хеши файлов. Если на Windows Git  
конвертирует LF → CRLF при checkout (поведение по умолчанию без `.gitattributes`),  
то содержимое файла на диске отличается от того, что было при вычислении хеша  
в `dvc.lock`. В результате DVC считает файл изменённым и перезапускает стадию,  
даже если содержимое данных не менялось. `.gitattributes` с `* text=auto eol=lf`  
фиксирует хранение файлов в LF, устраняя ложные изменения хешей.

### Почему метрики должны быть `cache: false`?

- С `cache: true` (по умолчанию) DVC хранит файл в `.dvc/cache` и создаёт  
  символическую ссылку в рабочей директории. Git не видит сам файл — только  
  ссылку в `dvc.lock`.  
- С `cache: false` файл остаётся в рабочей директории и коммитится напрямую  
  в Git. Это позволяет использовать `dvc metrics show`, `dvc metrics diff`  
  и сравнивать метрики между ветками/коммитами без настроенного DVC remote.  
  Преподаватель и ревьюер могут сразу видеть метрики в репозитории.

### Как DVC определяет, нужно ли перезапускать стадию?

При каждом `dvc repro` DVC:
1. Вычисляет MD5/SHA256 каждого файла в `deps` (код, данные).
2. Считывает текущие значения параметров из `params` (через YAML-ключи).
3. Сравнивает с записями в `dvc.lock`.
4. Если совпадает — пропускает стадию (skipping).
5. Если изменилось хотя бы что-то — перезапускает стадию и все стадии,  
   зависящие от её выходных файлов (каскадный пересчёт по DAG).

Это обеспечивает минимальное количество перезапусков при изменении части параметров.

---

## 7. Структура конвейера

```
validate ──→ preprocess ──→ split ──→ train ──→ evaluate
    │              │           │         │          │
 lr3.json   processed.csv  X_train  model.pkl  metrics.json
                          X_test
                          y_train
                          y_test
                          scaler.pkl
```

---

## 8. Тесты

```
$ pytest tests/test_pipeline.py -v

18 passed in 2.28s
```

**Ключевые тесты:**
- `test_no_arrdelay_in_features` — ArrDelay не в признаках (anti-leakage)
- `test_no_arrdelay_in_split_features` — ArrDelay не в X_train/X_test
- `test_label_is_binary` — метка строго 0/1
- `test_target_threshold_applied` — порог 15 применяется корректно
- `test_deterministic_split` — одинаковый seed даёт одинаковый сплит
- `test_metrics_has_required_keys` — metrics.json содержит все ключи
- `test_metrics_file_ends_with_newline` — файл заканчивается на `\n`

---

## 9. Проверка `git status` перед пушем

Файлы данных (`data/processed/`, `data/split/`, `models/`) исключены из Git  
через `.gitignore` и отслеживаются DVC через `dvc.lock`.

В Git коммитятся только:
- `dvc.yaml` — граф конвейера
- `dvc.lock` — зафиксированные хеши и параметры
- `params.yaml` — параметры
- `reports/metrics.json` — метрики (cache: false)
- `reports/validation/lr3.json` — отчёт валидации (cache: false)
- Код скриптов в `src/mlops_project/`
- Тесты в `tests/`
- Документация

---

_Evidence card заполнена: 2026-10-24. Тег `lr3-v1` и финальный SHA будут добавлены  
после слияния всех PR и выставления тега координатором @GlooshSl._
