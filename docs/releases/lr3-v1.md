# Релизная записка ЛР3: lr3-v1

Команда: 2
Репозиторий: https://github.com/kucheranton200/T26_MLOps
Контрольная точка: tag `lr3-v1`
Ответственный за выпуск: @GlooshSl
Срок сдачи по плану: пятница 23.10.2026, 14:00. Срок пропущен: тег и Release созданы после него.

## Что выпущено

Воспроизводимый конвейер DVC на данных ЛР2. Он запускается одной командой `dvc repro` и проходит пять стадий: `validate` → `preprocess` → `split` → `train` → `evaluate`. Параметры лежат в `params.yaml`, версии входов и выходов зафиксированы в `dvc.lock`, метрики в `reports/metrics.json`. Подробности плана и отклонений: `docs/plans/lr3-plan.md`.

## Данные

Файл `data/flights_sample.csv` с ЛР2 не менялся: 24 строки, 214 байт, SHA-256 `a828de7d3c87377c9bd247ae97ffc59ad90aa39dd43fa376380381c572268cbe`. Набор синтетический и лежит в Git, потому что он маленький. Контракт `data/contract.json` (версия 1.0.0) остаётся входным шлюзом конвейера: стадия `validate` проверяет данные до обработки. Описание набора: `docs/data-card.md`.

## Стадии конвейера

| Стадия | Команда | Выход |
|---|---|---|
| validate | `python -m mlops_project.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr3.json` | `reports/validation/lr3.json` (в Git) |
| preprocess | `python -m mlops_project.preprocess --config params.yaml` | `data/processed/flights_processed.csv`, `data/processed/preprocess_summary.json` |
| split | `python -m mlops_project.split_data --config params.yaml` | `data/split/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, `models/scaler.pkl` |
| train | `python -m mlops_project.train --config params.yaml` | `models/model.pkl` |
| evaluate | `python -m mlops_project.evaluate --config params.yaml` | `reports/metrics.json` (в Git) |

Ключевые параметры из `params.yaml`: цель `ArrDelay > 15`, признаки `DepHour` и `IsWeekend`, разбиение `test_size` 0,2, `random_state` 42, стратификация включена; модель `logistic_regression` с `C` 1,0 и `max_iter` 1000; порог оценки 0,5.

## Метрики базовой модели

| Метрика | Значение |
|---|---|
| roc_auc | 0,666667 |
| f1 | 0,5 |
| brier_score | 0,228108 |
| accuracy | 0,6 |
| n_test | 5 |

В обучении участвуют 19 строк, в тесте 5. На такой выборке метрики показывают только то, что конвейер работает от начала до конца, а не качество модели.

## Проверка воспроизводимости

Выполнено на Windows (PowerShell, Python 3.12.10, чистое дерево Git, `core.autocrlf=false`):

```
python -m pip install -e ".[dev,data]"
dvc repro
dvc repro
dvc status
python -m ruff check .
python -m pytest -q
```

Результат:
- первый `dvc repro`: стадия `validate` пропущена (её отчёт хранится в Git), стадии `preprocess`, `split`, `train`, `evaluate` выполнены заново, потому что обработанные данные и модель в Git не хранятся;
- второй `dvc repro`: пропущены все пять стадий, сообщение `Data and pipelines are up to date.`;
- `dvc status`: `Data and pipelines are up to date.`;
- `ruff`: замечаний нет; `pytest`: пройдено 29 тестов;
- хэши входов в `dvc.lock` (исходный код, параметры, данные) совпали с закоммиченными, метрики совпали с закоммиченными.

## Изменение параметра

В `params.yaml` значение `train.C` изменено с 1,0 на 0,5:
- `dvc status` показал изменение только в стадии `train` (раздел `train` в `params.yaml`);
- `dvc repro` пропустил `validate`, `preprocess`, `split` и выполнил `train`, `evaluate`;
- `dvc params diff`: `train.C` 1,0 → 0,5;
- `dvc metrics diff`: изменилась только `brier_score`, 0,22811 → 0,23111 (+0,003); `roc_auc`, `f1`, `accuracy` остались прежними.

После проверки значение возвращено командой `git restore`, изменения не коммитились.

## Что обнаружено при проверке

На Windows после `dvc repro` `git status` показывает изменённые `dvc.lock` и `reports/metrics.json`, хотя значения метрик и хэши входов совпадают с закоммиченными. Причины, насколько удалось установить:
- текстовые файлы, которые записывает конвейер, получают окончания строк CRLF в рабочей копии (`git ls-files --eol` показывает `w/crlf` при `i/lf` в репозитории), поэтому хэши таких выходов отличаются;
- хэши бинарных `models/model.pkl` и `models/scaler.pkl` тоже отличаются от закоммиченных. Предположительно из-за версий библиотек (у проверяющего scikit-learn 1.9.1 и numpy 2.5.4), но это не проверялось.

Файл `.gitattributes` (`* text=auto eol=lf`) задаёт LF для репозитория, но не влияет на файлы, которые Python записывает в рабочую копию. Это же описано в evidence card @Z0rko32. Решения на этот выпуск нет, проблема учитывается в ЛР4: новые стадии должны записывать файлы с LF.

## Утечки данных

По ревью @Vikabh (PR #52) и тестам `tests/test_pipeline.py`: `ArrDelay` служит только источником цели и в признаки не попадает; масштабирование обучается только на train.

## Ограничения

- Набор синтетический и маленький (24 строки, 5 в тесте), выводов о реальных рейсах не делается.
- Удалённое хранилище DVC не настроено (в ЛР3 по плану не требовалось), поэтому `dvc push` недоступен, а в чистом клоне нужные стадии выполняются заново.
- Версии библиотек в `pyproject.toml` не зафиксированы жёстко, это может влиять на хэши бинарных выходов.
- Конвейер построен на демонстрационном варианте (задержка рейсов). У команды вариант 10 «Отчисление студентов». Недостающее для варианта 10 из задач ЛР3 (две конфигурации признаков в `params.yaml`, сравнение между ними через `dvc metrics diff`, решение о конфигурации в ADR) выполняется в ЛР4 отдельным конвейером в папке `students/`. Конвейер ЛР3 и тег `lr3-v1` при этом не меняются.

## Отклонения от плана

- @Z0rko32 реализовал полный конвейер (скрипты, `dvc.yaml`, `dvc.lock`, `params.yaml`) одним PR вместо @kerty0.
- Добавлены стадии `train` и `evaluate`, которых в плане не было.
- Тег и Release созданы после срока сдачи.

Подробности и даты: раздел «Отклонения от плана» в `docs/plans/lr3-plan.md`.

## Задачи для ЛР4

- Сравнимые эксперименты в MLflow для варианта 10: baseline и две конфигурации признаков (ранний прогноз и поздний) на одной версии данных
- Основная метрика macro-F1, разброс оценивается бутстрепом
- План: `docs/plans/lr4-plan.md`

## Состав работы

| Часть | Автор | PR |
|---|---|---|
| План ЛР3 | @GlooshSl | #31 (issue #30) |
| Инициализация DVC | @kucheranton200 | #48 (issue #47) |
| Конвейер: стадии, `params.yaml`, `dvc.yaml`, `dvc.lock`, тесты | @Z0rko32 | #52 (issue #51) |
| Раздел карточки данных | @Vikabh | #50 (issue #49) |
| Шаг `dvc repro` в CI, evidence card | @kucheranton200 | #54 (issue #53) |
| README: команды DVC | @kerty0 | #56 (issue #55) |
| Evidence card | @Vikabh | #57 |
| Релизная записка, проверка воспроизводимости, evidence card, тег | @GlooshSl | этот PR (issue #60) |
