# Evidence card ЛР2

| Поле | Значение |
| --- | --- |
| **Студент** | Бахарева Виктория Юрьевна, группа T26УПМО-Т.МО41, команда team-2, GitHub: [@Vikabh](https://github.com/Vikabh) |
| **Роль** | Data steward / Documentation and risk reviewer — верификация набора данных, фиксация правил Data Leakage, рецензирование контракта, карточки данных и релизной документации |
| **Контрольная версия** | tag `lr2-v1`: https://github.com/kucheranton200/T26_MLOps/releases/tag/lr2-v1 |

## Собственный вклад (2-4 изменения)

| № | Что изменено | Issue | PR | Commit | Строки diff |
| - | --- | - | - | - | --- |
| 1 | **Документация датасета `docs/data-card.md`**: описание признаков `flights_sample.csv`, типов данных, диапазонов, правил предотвращения утечек данных (Data Leakage) | [#27](https://github.com/kucheranton200/T26_MLOps/issues/27) | [#26](https://github.com/kucheranton200/T26_MLOps/pull/26) | [`da2e9e7`](https://github.com/kucheranton200/T26_MLOps/commit/da2e9e7a4798f9ad34ad00dbf951c564649705b0) | [`docs/data-card.md`](https://github.com/kucheranton200/T26_MLOps/blob/main/docs/data-card.md) |
| 2 | **Персональная эвиденс-карточка `docs/evidence/lr2/lr2-evidence-card-Vikabh.md`**: подведение итогов роли Data steward, фиксация воспроизводимости, метрик валидатора и ревью | [#29](https://github.com/kucheranton200/T26_MLOps/issues/29) | [#34](https://github.com/kucheranton200/T26_MLOps/pull/34) | [`7af0b8f`](https://github.com/kucheranton200/T26_MLOps/commit/7af0b8f6f1b2e1d2d42f6dc8c32e7622136453ef) | [`docs/evidence/lr2/lr2-evidence-card-Vikabh.md`](https://github.com/kucheranton200/T26_MLOps/blob/main/docs/evidence/lr2/lr2-evidence-card-Vikabh.md) |

## Проверка коллеги

* **Ссылки на ревью**: 
  - [PR #41 (Release lead: итоговая релизная записка ЛР2)](https://github.com/kucheranton200/T26_MLOps/pull/41)
  - [PR #43 (Release lead: персональная эвиденс-карточка @GlooshSI)](https://github.com/kucheranton200/T26_MLOps/pull/43)
* **Проверенный критерий**: Верификация интеграции ссылки на карточку данных [`docs/data-card.md`](https://github.com/kucheranton200/T26_MLOps/blob/main/docs/data-card.md) и параметров предотвращения Data Leakage в итоговой релизной записке, а также проверка структуры персональной эвиденс-карточки координатора.

## Воспроизведение

* **Команда запуска валидатора**:
  `python -m mlops_project.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr2.json`
* **SHA-256 проверенного файла**: `a828de7d3c87377c9bd247ae97ffc59ad90aa39dd43fa376380381c572268cbe`
* **Результат успешной проверки (код завершения, `passed`):**
  * Код завершения: `0`
  * `passed`: `true`
  * Версия контракта: `1.0.0`
  * Датасет: `flight-delay-training-sample` (24 строки, 3 колонки: `DepHour`, `IsWeekend`, `ArrDelay`)
  * Проверки: Все 12 проверок пройдены со статусом `[PASS]`.

## Отрицательный сценарий

* **Какое нарушение я внёс самостоятельно (во временной копии):**
  1. Внесение недопустимого часа вылета `DepHour = 25` в `tmp_path`.
  2. Изменение значения признака выходного дня `IsWeekend = 2`.
  3. Удаление целевой колонки `ArrDelay` из датасета.
* **Какая проверка сработала и какой код завершения получен:**
  1. При `DepHour = 25`: сработала проверка `column_DepHour_range` (`passed: false`, `"Row 2: value 25 > maximum 23"`), код завершения `1`.
  2. При `IsWeekend = 2`: сработала проверка `column_IsWeekend_allowed_values` (`passed: false`, `"Row 2: value 2 not in [0, 1]"`), код завершения `1`.
  3. При отсутствии `ArrDelay`: сработала проверка `required_columns` (`passed: false`, `"Missing required columns: ['ArrDelay']"`), код завершения `1`.

## Обоснование

* **Правило контракта, которое я могу объяснить:**
  Диапазон `DepHour [0, 23]` и допустимые категории `IsWeekend [0, 1]` соответствуют жестким ограничениям предметной области (в сутках 24 часа, бинарный признак выходного дня). Диапазон `ArrDelay [-180, 1440]` описывает предельные задержки и опережения рейсов в минутах.
* **Риск утечки данных (Data Leakage):**
  Колонка `ArrDelay` предназначается строго для определения целевой переменной (target = ArrDelay > 15). Использование `ArrDelay` в списке входных признаков (FEATURES) приводит к Data Leakage — модели передается прямой ответ, что создаст иллюзию 100% точности на тренировочных данных, но сделает модель полностью неработоспособной в реальной эксплуатации.
* **Ограничение набора данных:**
  Датасет представляет собой учебную синтетическую выборку из 24 строк, предназначенную для отработки работоспособности валидаторов CI/CD и контрактов качества, а не для финального обучения продакшн-модели.

## Результат обучения

* Освоена концепция Data Contract как спецификации требований к качеству данных в MLOps.
* Изучены методы предотвращения утечек данных (Data Leakage) и их фиксация в документации датасета (`docs/data-card.md`).
* Закреплены навыки работы с Git-ветками, связыванием Issue и PR, оформлением документации и проведением содержательного Code Review в команде.

## Использование ИИ

* Помощь в структурировании карточки данных `data-card.md` и персональной эвиденс-карточки по стандарту команды.
* Формулирование и проверка правил предотвращения Data Leakage и рисков переобучения.
