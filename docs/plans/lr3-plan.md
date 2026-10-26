# План ЛР3: Предобработка данных и DVC-конвейер

Координатор (Release lead): @GlooshSl
Репозиторий команды: https://github.com/kucheranton200/T26_MLOps
Контрольная точка: tag `lr3-v1`
Срок сдачи: пятница 23.10.2026, 14:00
Основа: пошаговый пример преподавателя `МУ_ЛР-3.md`

Сроки ниже: внутренний план команды с запасом до срока сдачи.

## Что делаем (по примеру)

Превращаем разовые шаги ЛР2 в воспроизводимый конвейер DVC, который запускается одной командой `dvc repro`: проверка данных, предобработка, разделение на train и test. Параметры лежат в `params.yaml`, версии зафиксированы в `dvc.lock`.

Не входит в ЛР3: общее удалённое хранилище (remote), эксперименты `dvc exp`, метрики и стадия обучения модели. В примере преподавателя их нет.

Требования ЛР2 сохраняются: валидатор и контракт остаются входным шлюзом конвейера, `ArrDelay` не входит в признаки, преобразования обучаются только на train.

## Роли и основание выбора

| Роль | Ответственный | Основание выбора (роль в ЛР1 и ЛР2) |
|---|---|---|
| Quality engineer | @Z0rko32 | Quality engineer в ЛР1 и ЛР2: код, тесты |
| Environment engineer | @kerty0 | Environment engineer в ЛР1: зависимости, README, настройка окружения |
| Repository administrator | @kucheranton200 | Repository administrator: структура репозитория, CI, чистый клон |
| Documentation and risk reviewer | @Vikabh | Documentation and risk reviewer: карточка данных, риски, ревью утечки |
| Координатор (Release lead) | @GlooshSl | Release lead: план, связи issue и PR, тег и релиз |

## Задачи

| Задача | Ответственный | Файлы | Issue | Ревью | Срок |
|---|---|---|---|---|---|
| План ЛР3 | @GlooshSl | docs/plans/lr3-plan.md | #30 | @kucheranton200, @Z0rko32 | пн 19.10 |
| Инициализация DVC (`dvc init`), решение по исходному файлу данных, .gitignore и .dvcignore | @kucheranton200 | .dvc/, .dvcignore, .gitignore | #? | @kerty0 | вт 20.10, 12:00 |
| Зависимость dvc в проекте и params.yaml | @kerty0 | pyproject.toml, params.yaml | #? | @kucheranton200 | вт 20.10, 12:00 |
| Предобработка: очистка, целевая переменная `ArrDelay > 15` | @Z0rko32 | src/mlops_project/preprocess.py | #? | @Vikabh, @kerty0 | вт 20.10, 20:00 |
| Разделение train и test (стратификация, random_state=42, масштабирование только на train) | @Z0rko32 | src/mlops_project/split_data.py | #? | @Vikabh, @kerty0 | ср 21.10, 12:00 |
| Тесты стадий (в том числе против утечки: `ArrDelay` не в признаках) | @Z0rko32 | tests/test_pipeline.py | #? | @Vikabh | ср 21.10, 12:00 |
| Конвейер: dvc.yaml и dvc.lock (после слияния скриптов) | @kerty0 | dvc.yaml, dvc.lock | #? | @Z0rko32 | ср 21.10, 18:00 |
| Шаг `dvc repro` в CI | @kucheranton200 | .github/workflows/ci.yml | #53 | @Z0rko32 | ср 21.10, 21:00 |
| Раздел про DVC в карточке данных: версионирование, обновление, риски, ревью утечки | @Vikabh | docs/data-card.md | #? | @kucheranton200 | чт 22.10, 12:00 |
| README: установка DVC, `dvc repro`, команды для Windows | @kerty0 | README.md | #? | @Vikabh | чт 22.10, 12:00 |
| Релизная записка ЛР3 | @GlooshSl | docs/releases/lr3-v1.md | #? | @kucheranton200 | чт 22.10, 22:00 (черновик), пт 23.10, 10:00 (итог) |
| Ревью всех PR (содержательное, с запуском проверок) | вся команда | - | - | - | чт 22.10, до 18:00 |
| Tag lr3-v1 и Release | @GlooshSl | - | #? | - | пт 23.10, 09:30 |
| Проверка по тегу на чистом клоне: `dvc repro` с нуля | @kucheranton200 | - | - | - | пт 23.10, 10:30 |
| Evidence card каждого участника | каждый свою | docs/evidence/lr3-<логин>.md | #? | - | пт 23.10, 13:00 |
| Ссылки в ДО.СКИФ | по правилам команды | - | - | - | пт 23.10, до 14:00 |

## Договорённости по интерфейсу (чтобы работать параллельно)

| Стадия | Вход | Выход | Команда |
|---|---|---|---|
| validate | data/flights_sample.csv, data/contract.json | reports/validation/lr3.json | `python -m mlops_project.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr3.json` |
| preprocess | data/flights_sample.csv, params.yaml | data/processed/flights_processed.csv | `python -m mlops_project.preprocess --config params.yaml` |
| split | data/processed/flights_processed.csv, params.yaml | data/split/X_train.csv, X_test.csv, y_train.csv, y_test.csv, models/scaler.pkl | `python -m mlops_project.split_data --config params.yaml` |

Параметры в `params.yaml`: `test_size`, `random_state` (42) и правила предобработки. Точные пути модулей подтверждает @Z0rko32 по фактической структуре `src/mlops_project`.

## График

| Когда | Что должно быть готово |
|---|---|
| Пн 19.10, вечер | План влит, у каждого создана своя issue |
| Вт 20.10, 12:00 | PR с DVC (@kucheranton200) и PR с зависимостью и params.yaml (@kerty0) |
| Вт 20.10, 20:00 | PR с предобработкой (@Z0rko32) |
| Ср 21.10, 12:00 | PR с разделением и тесты (@Z0rko32) |
| Ср 21.10, 18:00 | PR с dvc.yaml и dvc.lock (@kerty0), после слияния скриптов |
| Ср 21.10, 21:00 | PR с шагом CI (@kucheranton200) |
| Чт 22.10, 12:00 | PR с разделом карточки данных (@Vikabh), PR с README (@kerty0) |
| Чт 22.10, до 18:00 | Ревью всех PR, CI зелёный, всё влито |
| Пт 23.10, 09:30 | Tag lr3-v1 и Release с записками |
| Пт 23.10, 10:30 | Проверка по тегу на чистом клоне |
| Пт 23.10, до 13:00 | В evidence card дописаны тег и SHA |
| Пт 23.10, 14:00 | Срок сдачи |

Порядок слияния важен: DVC и params (вт), скрипты (вт и ср), dvc.yaml с dvc.lock (ср, после скриптов), CI (ср вечер), документы (чт).

## Решения команды

1. **Исходный файл данных.** `data/flights_sample.csv` весит 214 байт и уже лежит в Git. Предлагаем оставить его в Git и указать в `dvc.yaml` как зависимость, не переводя под `dvc add`: так CI и чистый клон получают данные без удалённого хранилища. Выходы стадий DVC отслеживает сам. Решение фиксирует @kucheranton200 и записывает в карточку данных.
2. **Масштабирование.** Если используется, обучается только на train и сохраняется в models/scaler.pkl (правило ЛР2).
3. **Отчёт валидации** для CI формируется стадией validate и фиксируется в `dvc.lock`.

## Порядок работы

1. У каждого участника своя issue с наблюдаемым результатом и критериями приёмки.
2. Работа идёт в ветках `lab3/<название>` от свежего main. В main только через pull request, в описании `Closes #N`.
3. Merge после зелёного CI и одобрения другого участника.
4. Tag lr3-v1 ставится после merge всех pull request.
5. Evidence card дописывается ссылкой на тег и SHA после его создания.

## Зависимости и риски

- Нужны скрипты до `dvc.yaml`: иначе `dvc repro` нечем запускать. Поэтому dvc.yaml сливается после preprocess и split.
- `dvc.lock` создаётся запуском `dvc repro` и коммитится вручную: без него конвейер не воспроизводится.
- Команды примера написаны для Linux. На Windows их нужно переводить: `mkdir -p`, `sed -i`, `source venv/bin/activate` не работают в PowerShell.
- Не выполнять `dvc install` без настроенного remote: хук `pre-push` будет запускать `dvc push` и ломать `git push`.
- Утечка из примера: в нём признаки содержат поля, из которых вычисляется цель, и масштабирование идёт до разделения. У нас так нельзя (правило ЛР2).
- Нет ответа от участника до срока: задача переходит координатору или @kucheranton200 по договорённости.
- Команда отстаёт от графика (ЛР2): после слияния плана ЛР3 сначала закрыть остатки ЛР2.

## Отклонения от плана

Сюда записываются передача и изменение задач по договорённости в ходе работы. Строки основной таблицы не переписываются: так видно, что было запланировано и что изменилось.

| Дата | Задача | Было | Стало | Причина | Кто согласовал |
|---|---|---|---|---|---|
| 2026-10-24 | Конвейер: dvc.yaml и dvc.lock | @kerty0, ср 21.10 | @Z0rko32, 24.10 | @kerty0 не успел; @Z0rko32 реализовал полный конвейер (скрипты + dvc.yaml + dvc.lock + params.yaml) в рамках одного PR | @Z0rko32 |
| 2026-10-24 | Стадии train и evaluate | Не входило в ЛР3 по плану | Добавлены как baseline | Для полноты воспроизводимого конвейера и демонстрации метрик; стадии опциональны, но полезны | @Z0rko32 |
