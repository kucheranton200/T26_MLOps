# Evidence card ЛР2

| Поле | Значение |
|---|---|
| Студент | Кныш Радион, группа Т26УПМО-Т.МО41, команда team-2, GitHub: @Z0rko32 |
| Роль | Quality engineer: разработка валидатора данных, контракта, тестов и интеграция в CI |
| Контрольная версия | tag `lr2-v1`: `<ссылка на tag>`, commit SHA: `<SHA>` |

## Собственный вклад (2-4 изменения)

| № | Что изменено | Issue | PR | Commit | Строки diff |
|---|---|---|---|---|---|
| 1 | Валидатор `src/mlops_project/validate_data.py` и модуль совместимости `src/mlops_flight_delay/validate_data.py`: полная валидация схемы, обязательности, типов (целочисленность), диапазонов, допустимых значений, дубликатов, расчет SHA-256 и генерация JSON-отчета | #37 | #38 | [1d875a1](https://github.com/kucheranton200/T26_MLOps/commit/1d875a176d0d358bc9818e809648077db7b7457b) | `src/mlops_project/validate_data.py`, `src/mlops_flight_delay/validate_data.py` |
| 2 | Автотесты контракта данных `tests/test_data_contract.py`: позитивный тест на эталонном `data/flights_sample.csv` и комплексные негативные сценарии в `tmp_path` (выход за диапазон, недопустимое категориальное значение, отсутствие колонки, дубликаты, недостаток строк) | #37 | #38 | [1d875a1](https://github.com/kucheranton200/T26_MLOps/commit/1d875a176d0d358bc9818e809648077db7b7457b) | `tests/test_data_contract.py` |
| 3 | Интеграция шага валидации данных в GitHub Actions CI `.github/workflows/ci.yml` перед запуском автотестов | #37 | #38 | [1d875a1](https://github.com/kucheranton200/T26_MLOps/commit/1d875a176d0d358bc9818e809648077db7b7457b) | `.github/workflows/ci.yml` |
| 4 | Синхронизация контракта `data/contract.json` (диапазон ArrDelay, minimum_rows) и фиксация статуса задач в документации (`docs/plans/lr2-plan.md`, `docs/team-roles.md`) | #37 | #38 | [1d875a1](https://github.com/kucheranton200/T26_MLOps/commit/1d875a176d0d358bc9818e809648077db7b7457b) | `data/contract.json`, `docs/plans/lr2-plan.md`, `docs/team-roles.md` |

## Проверка коллеги

Проверка и ревью PR #25 / PR #36 (контракт данных `data/contract.json` от @kerty0, ревью @GlooshSl):
- Ссылка на ревью: https://github.com/kucheranton200/T26_MLOps/pull/25#pullrequestreview-5582988947 и https://github.com/kucheranton200/T26_MLOps/pull/36#pullrequestreview-5584625048
- Проверенный критерий: верификация наличия колонки `ArrDelay` в контракте, корректность задания ее типа (`integer`), диапазона (`-180..1440`) и роли (`target_source`), что предотвращает ложное падение проверки эталонного файла при флаге `allow_extra_columns: false`.

## Воспроизведение

- Команда запуска валидатора:
  ```bash
  python -m mlops_project.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr2.json
  ```
  *(также поддерживается вызов: `python -m mlops_flight_delay.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr2.json`)*
- SHA-256 проверенного файла: `a828de7d3c87377c9bd247ae97ffc59ad90aa39dd43fa376380381c572268cbe`
- Результат успешной проверки:
  - Код завершения: `0`
  - `passed`: `true`
  - Версия контракта: `1.0.0`
  - Датасет: `flight-delay-training-sample`
  - Число строк: `24`
  - Колонки: `["DepHour", "IsWeekend", "ArrDelay"]`
  - Результаты проверок (`checks`):
    - `required_columns`: `[PASS]` All 3 required columns present
    - `no_extra_columns`: `[PASS]` No unexpected extra columns
    - `minimum_rows`: `[PASS]` Row count 24 meets minimum of 20
    - `no_duplicate_rows`: `[PASS]` All 24 rows are unique
    - `column_DepHour_nullability`: `[PASS]` No null values
    - `column_DepHour_type`: `[PASS]` All values match type 'integer'
    - `column_DepHour_range`: `[PASS]` All values within [0, 23]
    - `column_IsWeekend_nullability`: `[PASS]` No null values
    - `column_IsWeekend_type`: `[PASS]` All values match type 'integer'
    - `column_IsWeekend_allowed_values`: `[PASS]` All values in [0, 1]
    - `column_ArrDelay_nullability`: `[PASS]` No null values
    - `column_ArrDelay_type`: `[PASS]` All values match type 'integer'
    - `column_ArrDelay_range`: `[PASS]` All values within [-180, 1440]

## Отрицательный сценарий

- Нарушения, внесенные во временной копии (`tmp_path`) без изменения эталонного файла:
  1. Значение `DepHour = 25` (нарушение верхней границы диапазона [0, 23]).
  2. Значение `IsWeekend = 2` (нарушение списка допустимых категорий [0, 1]).
  3. Удаление колонки `ArrDelay` (нарушение обязательного состава колонок).
- Сработавшие проверки и коды завершения:
  1. Сценарий `DepHour = 25`: сработала проверка `column_DepHour_range` (`passed: false`, `Row 2: value 25 > maximum 23`), код завершения `1`.
  2. Сценарий `IsWeekend = 2`: сработала проверка `column_IsWeekend_allowed_values` (`passed: false`, `Row 2: value 2 not in [0, 1]`), код завершения `1`.
  3. Сценарий отсутствия `ArrDelay`: сработала проверка `required_columns` (`passed: false`, `Missing required columns: ['ArrDelay']`), код завершения `1`.

## Обоснование

- **Правило контракта:** границы `DepHour` [0, 23] и допустимые значения `IsWeekend` [0, 1] следуют из предметной логики (в сутках 24 часа, день либо выходной, либо рабочий). Диапазон `ArrDelay` [-180, 1440] соответствует возможным задержкам рейса в минутах (до 3 часов опережения и до 24 часов опоздания).
- **Риск утечки данных (Data Leakage):** колонка `ArrDelay` используется исключительно для разметки целевой переменной (`target = ArrDelay > 15`). Использование `ArrDelay` в качестве признака для обучения модели (`FEATURES`) категорически недопустимо, так как это прямая утечка ответа, приводящая к фиктивному 100% качеству при обучении и полной неработоспособности модели в реальной эксплуатации, когда рейс еще не прилетел.
- **Стратегия разделения данных:** случайное стратифицированное разделение (`random_state=42`) с сохранением соотношения классов целевой переменной (задержан/не задержан) в обучающей и валидационной выборках. Любые масштабирования и предобработка обучаются строго на train-выборке.
- **Ограничение набора данных:** датасет является синтетическим учебным набором фиксированного размера (24 строки), предназначенным для тестирования пайплайна и отработки контрактов качества, а не для аналитических выводов о реальном авиасообщении.

## Результат обучения

- Освоена концепция Data Contract как машиночитаемой спецификации требований к качеству данных в MLOps.
- Реализована привязка конкретного состояния файла данных к валидационному отчету через SHA-256 хеш.
- Отработана практика изолированного тестирования негативных сценариев с использованием фикстуры `tmp_path` в `pytest`.
- Настроен пайплайн CI, гарантирующий блокировку интеграции при несоответствии входных данных контракту.

## Использование ИИ

- Помощь в формулировании архитектуры валидатора данных и покрытия граничных условий (строгая типизация целых чисел, обработка float/пустых строк).
- Генерация шаблонов негативных автотестов в pytest.
- Код валидатора и тестов проверен локальным запуском `ruff check` и `pytest`, все проверки пройдены со статусом 100%.
