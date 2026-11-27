# Evidence Card ЛР4 — Кныш Радион (@Z0rko32)

**Роль в ЛР4:** Quality engineer  
**Команда:** team-2  
**Репозиторий:** https://github.com/kucheranton200/T26_MLOps  
**Ветка:** `lab4/quality-zorko32`  
**Базовый коммит main:** `89e9b3c`  
**Issue / PR:** [#72](https://github.com/kucheranton200/T26_MLOps/issues/72)  
**Тег:** `lr4-v1` _(ставится после слияния всех PR по ЛР4)_

---

## 1. Выполненные задачи (роль Quality engineer)

В соответствии с [`docs/plans/lr4-plan.md`](../../plans/lr4-plan.md) реализованы стадии конвейера `students/` (вариант 10, «Отчисление студентов») и автотесты к ним, без изменения корневых файлов ЛР3 и без вмешательства в задачи других участников:

| Артефакт | Назначение |
|---|---|
| `src/mlops_project/students/__init__.py` | Инициализация пакета стадий ЛР4 |
| `src/mlops_project/students/preprocess.py` | Отбор признаков `early` (6 признаков) или `late` (12 признаков) + `Target`; защита от попадания `Target` в признаки |
| `src/mlops_project/students/split_data.py` | Стратифицированное разбиение по `Target` (`test_size=0.2`, `random_state=42`); без скейлинга (скейлер внутри `Pipeline` в `train.py`) |
| `src/mlops_project/students/train.py` | Обучение `majority` (`DummyClassifier`) и `logreg` (`StandardScaler` + `LogisticRegression(class_weight="balanced")`); сериализация в `skops` (без `pickle`); ленивый импорт MLflow только при `--mlflow` (теги `git_commit`, `data_sha256`, `contract_version`, `prediction_moment`, `feature_set`, сигнатура и артефакт) |
| `src/mlops_project/students/evaluate.py` | Расчёт `macro_f1`, `f1_dropout`, `f1_enrolled`, `f1_graduate`, `weighted_f1`; сохранение `metrics.json` и `test_predictions.csv`; дозапись метрик в тот же MLflow run при `--mlflow` |
| `tests/test_students_pipeline.py` | 5 тестов на `preprocess` и `split_data` (ранняя/поздняя конфигурации, запрет `Target` в признаках, стратификация, детерминизм) |
| `tests/test_students_train.py` | 6 тестов на `train` и `evaluate` (`majority` и `logreg`, расчёт метрик, работа без MLflow, roundtrip `skops` и MLflow при наличии зависимостей) |
| `docs/team-roles.md` | Добавлена строка роли `@Z0rko32` для ЛР4 |

---

## 2. Локальная проверка и CI

```powershell
python -m ruff check .
python -m pytest -q
```

- `ruff check .` — `All checks passed!`
- `pytest -q` — `38 passed, 2 skipped` (2 интеграционных теста `test_real_skops_roundtrip` и `test_real_mlflow_logging` корректно пропускаются через `pytest.importorskip`, так как зависимости `skops` и `mlflow` из задачи `extra tracking` ещё не влиты в `main`).

---

## 3. Ограничения и зависимости от задач других участников

- В `main` на момент выполнения работы ещё не влиты: `data/students_sample.csv` (открытый WIP PR #66), `extra tracking` в `pyproject.toml` (`mlflow`, `skops`), `students/params.yaml` и `students/dvc.yaml`.
- По этой причине три живых запуска в `mlflow.db` и прогон `dvc repro students/dvc.yaml` выполняются после слияния указанных зависимостей; все тесты в `tests/test_students_*.py` используют изолированные синтетические фикстуры во временной директории (`tmp_path`) и не затрагивают чужие файлы.

---

## 4. Использование ИИ

При подготовке кода модулей `src/mlops_project/students/*`, автотестов `tests/test_students_*.py` и оформлении карточки использовался ИИ-ассистент; весь код проверен линтером `ruff` и тестами `pytest`.
