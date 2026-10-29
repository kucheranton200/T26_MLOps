# План ЛР4: Сравнимые эксперименты в MLflow (вариант 10, «Отчисление студентов»)

Координатор (Release lead): @GlooshSl
Репозиторий команды: https://github.com/kucheranton200/T26_MLOps
Контрольная точка: tag `lr4-v1`
Срок в ДО.СКИФ: пятница 30.10.2026, 00:00. Команда сдвигает внутренний план на неделю; срок команды: четверг 05.11.2026. Сдвиг и репозиторий сдачи (страница сдачи ссылается на Irina-64/T25--4m_MLOps) нужно подтвердить у преподавателя.
Основа: методичка «Лабораторная работа 4. Сравнимые эксперименты в MLflow» и задание трека 10.

## Что делаем

Сравниваем варианты модели так, чтобы результат можно было повторить, а выбор обосновать. Фактор эксперимента: конфигурация признаков.

- **baseline**: константный прогноз мажоритарного класса (Graduate). Точка отсчёта macro-F1 = 2p/(1+p)/3, при p = 0,573 это ≈ 0,243.
- **вариант 1, ранняя конфигурация** (`prediction_moment=enrollment`): Age, Gender, DaytimeAttendance, Scholarship, PrevGrade, AdmissionGrade.
- **вариант 2, поздняя конфигурация** (`prediction_moment=after_sem2`): ранняя плюс Sem1Enrolled, Sem1Approved, Sem1Grade, Sem2Enrolled, Sem2Approved, Sem2Grade.

Основная метрика: macro-F1. Дополнительные: F1 класса Dropout, F1 каждого класса. weighted-F1 логируется только как справочная и в выборе не участвует: она скрыла бы слабость на классе Enrolled.

Данные: синтетический `data/students_sample.csv` (1200 строк, все столбцы целочисленные, Target: 0 Dropout 28,4 %, 1 Enrolled 14,3 %, 2 Graduate 57,3 %), SHA-256 `3128fe6b8c6e7cd67d34597fe14b87312a56461a2531585d8f6bf285e849eadb`, 39292 байта. Данные синтетические, это прямо пишется в README и карточке данных. Все запуски идут на одной версии данных: тег `data_sha256` одинаковый.

Вывод в отчёте разделяется на два: «лучшее качество» (скорее всего поздняя конфигурация) и «пригодно для раннего сигнала» (только то, что известно на момент зачисления). Поздний прогноз может выглядеть лучше, но быть бесполезным для куратора.

Не входит в ЛР4: реестр моделей и решение о выпуске (ЛР5), `dvc exp`, мониторинг.

## Границы работы: вариант 10 строится рядом с ЛР3

ЛР3 завершена. Её тег `lr3-v1` и файлы мы не трогаем и шагов по ЛР3 в этом плане нет. Для варианта 10 строится отдельный конвейер в папке `students/`. Что для варианта 10 в ЛР3 не было сделано (две конфигурации признаков в `params.yaml`, сравнение через `dvc metrics diff`, решение о конфигурации в ADR), закрывается внутри этого конвейера и прямо записывается в релизной записке `lr4-v1`.

| Категория | Что |
|---|---|
| Не меняем | корневые `dvc.yaml`, `dvc.lock`, `params.yaml`; модули `src/mlops_project/` (кроме новой подпапки `students/`); существующие тесты; `data/contract.json`; `data/flights_sample.csv`; существующие шаги CI; тег `lr3-v1` |
| Меняем только добавлением | `pyproject.toml` (extra `tracking`), `.gitignore`, `.github/workflows/ci.yml` (установка extra и шаг для `students/`), `README.md` (новые разделы), `docs/data-card.md` (новый раздел), `docs/adr/` |
| Новое | `students/` (`params.yaml`, `dvc.yaml`, `dvc.lock`, выходы стадий), `src/mlops_project/students/`, `data/students_sample.csv`, `data/students_contract.json`, `scripts/generate_students.py`, `scripts/compare_runs.py`, `tests/test_students_*.py` |

Модуль `mlops_project.validate_data` переиспользуется без правок. Если он не справляется с контрактом студентов, добавляется `src/mlops_project/students/validate.py` (задача про контракт), а запись об этом вносится в раздел «Отклонения».

## Роли и основание выбора

| Роль | Ответственный | Основание выбора |
|---|---|---|
| Release lead (координатор) | @GlooshSl | планы, связь issue и PR, записки, теги, evidence cards; набор данных, отчёт сравнения конфигураций |
| Repository administrator | @kucheranton200 | репозиторий, CI, `.gitignore`, ветка `lr4-base`, чистый клон; скрипт сравнения, повтор лучшего запуска |
| Environment engineer | @kerty0 | зависимости, окружение, README; extra `tracking`, контракт, `params.yaml` и `dvc.yaml` |
| Quality engineer | @Z0rko32 | код, тесты; стадии конвейера, обучение, оценка, журналирование в MLflow, запуски эксперимента |
| Documentation and risk reviewer | @Vikabh | ADR, карточка данных, риски; ADR 0002, ADR о конфигурации, таблица сравнения в README |

## Задачи

В колонке «Issue» заголовок issue; номер вписывается в описание PR после создания. Ветки `lab4/<название>` от свежего main.

| Задача | Ответственный | Файлы | Issue | Ревью | Срок | Готово, когда |
|---|---|---|---|---|---|---|
| Ветка `lr4-base` | @kucheranton200 | - (ветка в репозитории) | ЛР4: ветка lr4-base | - | чт 29.10, до слияния плана | ветка создана от main до слияния PR плана, не удаляется и не двигается |
| План ЛР4 | @GlooshSl | docs/plans/lr4-plan.md | ЛР4: план | @kucheranton200, @Z0rko32 | чт 29.10, вечер | PR влит, у каждого создана своя issue |
| Генератор и набор данных | @GlooshSl | scripts/generate_students.py, data/students_sample.csv | ЛР4: генератор и набор «Отчисление студентов» | @Z0rko32, @Vikabh | пт 30.10, 18:00 | SHA-256 файла совпадает с указанным выше, генератор воспроизводит файл |
| `.gitignore` для MLflow | @kucheranton200 | .gitignore | ЛР4: .gitignore для MLflow | @kerty0 | пн 02.11, 12:00 | исключены `mlflow.db`, `mlruns/`, `mlartifacts/`, `students/reports/run.json` |
| Extra `tracking` | @kerty0 | pyproject.toml | ЛР4: extra tracking | @kucheranton200 | пн 02.11, 12:00 | `pip install -e ".[dev,data,tracking]"` ставит MLflow 3.16, scikit-learn 1.9, skops |
| CI: установка `.[dev,data,tracking]` | @kucheranton200 | .github/workflows/ci.yml | ЛР4: CI, установка tracking | @Z0rko32 | пн 02.11, 18:00 | CI зелёный, pytest импортирует mlflow |
| Контракт v2.0.0 и тесты | @kerty0 | data/students_contract.json, tests/test_students_contract.py (при необходимости src/mlops_project/students/validate.py) | ЛР4: контракт v2.0.0 | @Z0rko32, @Vikabh | пн 02.11, 18:00 | валидатор принимает набор, отклоняет нарушенный (Target вне 0–2, пропуск, лишний столбец); решение о модуле валидации описано в PR |
| ADR 0002 «основная метрика и baseline» | @Vikabh | docs/adr/0002-main-metric-and-baseline.md | ЛР4: ADR 0002 | @GlooshSl, @Z0rko32 | вт 03.11, 12:00 | зафиксированы macro-F1, точка отсчёта ≈ 0,243, фактор, фиксированные условия, правило признания различия, skops без pickle |
| ADR о конфигурации признаков в main | @Vikabh | docs/adr/0001-*.md или docs/adr/0003-*.md | ЛР4: ADR о конфигурации признаков | @kucheranton200 | вт 03.11, 12:00 | если ADR 0001 посвящён этому вопросу, он дополняется; если другому, пишется ADR 0003, существующие ADR не переписываются |
| Стадии preprocess, split и тесты | @Z0rko32 | src/mlops_project/students/__init__.py, preprocess.py, split_data.py, tests/test_students_pipeline.py | ЛР4: preprocess и split | @Vikabh, @kerty0 | вт 03.11, 12:00 | стратифицированное разбиение воспроизводимо, Target не попадает в признаки |
| Стадии train и evaluate, журналирование в MLflow | @Z0rko32 | src/mlops_project/students/train.py, evaluate.py, tests/test_students_train.py | ЛР4: train, evaluate, MLflow | @kucheranton200, @kerty0 | вт 03.11, 20:00 | с `--mlflow` создаётся запуск с тегами, метриками, сигнатурой, skops; без флага MLflow не используется |
| Конвейер `students/`: параметры, `dvc.yaml`, `dvc.lock`, шаг CI | @kerty0 | students/params.yaml, students/dvc.yaml, students/dvc.lock, students/reports/metrics.json, .github/workflows/ci.yml (один шаг) | ЛР4: params.yaml и dvc.yaml | @Z0rko32, @kucheranton200 | ср 04.11, 10:00 | `dvc repro students/dvc.yaml` дважды: второй раз все стадии пропущены; смена `features.config` не перезапускает validate |
| Раздел о новом наборе в карточке данных | @Vikabh | docs/data-card.md | ЛР4: карточка данных | @kucheranton200 | ср 04.11, 12:00 | указаны источник (синтетика), SHA-256, классы, риски |
| Скрипт сравнения запусков и бутстреп | @kucheranton200 | scripts/compare_runs.py, tests/test_compare_runs.py | ЛР4: скрипт сравнения и бутстреп | @Z0rko32 | ср 04.11, 14:00 | `mlflow.search_runs` и 1000 повторов по тесту дают таблицу в Markdown с 95 %-интервалами и парной разностью |
| Три запуска: baseline, ранняя, поздняя | @Z0rko32 | - (`mlflow.db` не коммитится) | ЛР4: запуски эксперимента | @kucheranton200 | ср 04.11, 16:00 | три запуска в `student-dropout`, одинаковый `data_sha256`; вывод скрипта сравнения приложен в issue |
| Отчёт `dvc metrics diff` между конфигурациями | @GlooshSl | docs/reports/feature-configs.md | ЛР4: отчёт dvc metrics diff | @Vikabh | ср 04.11, 20:00 | в отчёте цена раннего прогноза и вывод «пригодно для раннего сигнала» отдельно |
| README: команды DVC и MLflow | @kerty0 | README.md (раздел «Конвейер students и эксперименты») | ЛР4: README, команды | @Vikabh | чт 05.11, 10:00 | команды из раздела выполняются на чистом клоне |
| README: таблица сравнения | @Vikabh | README.md (раздел «Результаты сравнения») | ЛР4: таблица сравнения | @GlooshSl, @kucheranton200 | чт 05.11, 10:00 | точка отсчёта, разброс, число попыток и два вывода |
| Повтор лучшего запуска на чистом клоне | @kucheranton200 | - | ЛР4: повтор лучшего запуска | - | чт 05.11, 12:00 | macro-F1 совпадает с таблицей; `dvc status` пуст; `git ls-files --eol` без CRLF; результат комментарием в issue |
| Ревью всех PR (содержательное, с запуском проверок) | вся команда | - | - | - | чт 05.11, до 15:00 | CI зелёный, всё влито |
| Записка, карточка, отчёт, отклонения в плане | @GlooshSl | docs/releases/lr4-v1.md, docs/evidence/lr4-GlooshSl.md, docs/plans/lr4-plan.md | ЛР4: записка и карточка | @kucheranton200 | чт 05.11, 16:00 | в записке указано, что недостающее для варианта 10 из ЛР3 закрыто отдельным конвейером |
| Evidence card ЛР4 (у каждого своя) | @kerty0, @Z0rko32, @Vikabh, @kucheranton200 | docs/evidence/lr4-<логин>.md | ЛР4: evidence card (у каждого своя) | другой участник | чт 05.11, 17:00 | влиты до 18:00, использование ИИ указано честно |
| Tag `lr4-v1` и GitHub Release | @GlooshSl | - | ЛР4: тег и Release | - | чт 05.11, 18:00 | тег на main после влития всех PR, не переставляется |
| Сводные PR для ДО.СКИФ (не вливаются) | @GlooshSl, @kerty0, @Z0rko32, @Vikabh | - | ЛР4: сводный PR (у каждого свой) | - | чт 05.11, 19:30 | PR из своей ветки в `lr4-base` со всеми своими коммитами и таблицей |
| Ссылки в ДО.СКИФ | четверо: свой сводный PR; @kucheranton200: ссылка на main | - | - | - | чт 05.11, 20:00 | ссылки сданы |

## Договорённости по интерфейсу для параллельной работы

Пути указаны от корня репозитория. Команды запускаются из корня. Выходы стадий лежат в `students/`, чтобы не пересекаться с конвейером ЛР3.

| Стадия | Вход | Выход | Команда |
|---|---|---|---|
| validate | data/students_sample.csv, data/students_contract.json | students/reports/validation.json | `python -m mlops_project.validate_data --data data/students_sample.csv --contract data/students_contract.json --report students/reports/validation.json` |
| preprocess | data/students_sample.csv, students/params.yaml (раздел `features`) | students/data/processed/students.csv | `python -m mlops_project.students.preprocess --config students/params.yaml` |
| split | students/data/processed/students.csv, students/params.yaml (раздел `split`) | students/data/split/X_train.csv, X_test.csv, y_train.csv, y_test.csv | `python -m mlops_project.students.split_data --config students/params.yaml` |
| train | students/data/split/X_train.csv, y_train.csv, students/params.yaml (раздел `train`) | students/models/model.skops; students/reports/run.json (только при `--mlflow`, в Git не коммитится) | `python -m mlops_project.students.train --config students/params.yaml` |
| evaluate | students/models/model.skops, students/data/split/X_test.csv, y_test.csv | students/reports/metrics.json (в Git, `cache: false`), students/data/eval/test_predictions.csv | `python -m mlops_project.students.evaluate --config students/params.yaml` |

Стадия preprocess отбирает столбцы по `features.config` и Target. Поэтому смена конфигурации перезапускает preprocess и всё после него, но не validate.

Параметры в `students/params.yaml` (названия разделов фиксируем сейчас, значения подтверждает @kerty0):

- `features.config`: `early` или `late`; `features.early` и `features.late`: списки столбцов, как в разделе «Что делаем»;
- `split.test_size` (0,2), `split.random_state` (42), разбиение стратифицированное по Target;
- `train.model`: `majority` или `logreg`, `train.logreg_c` (1,0), `train.seed` (42);
- `mlflow.experiment`: `student-dropout`.

Правила журналирования (одинаковы для всех запусков):

- train с `--mlflow` создаёт запуск и пишет `students/reports/run.json` с идентификатором; evaluate с `--mlflow` дописывает метрики в тот же запуск и логирует `test_predictions.csv` как артефакт (из него скрипт сравнения считает бутстреп);
- имя запуска содержит фактор: `baseline-majority-seed42`, `logreg-early-seed42`, `logreg-late-seed42`;
- теги: `git_commit`, `data_sha256` (считается из файла, не вводится вручную), `contract_version`, `prediction_moment` (`none` для baseline, `enrollment`, `after_sem2`), `feature_set` (`none`, `early`, `late`);
- метрики: `macro_f1`, `f1_dropout`, `f1_enrolled`, `f1_graduate`, `weighted_f1` (справочно);
- модель с сигнатурой и входным примером, формат skops, pickle не используется;
- без `--mlflow` конвейер DVC работает без MLflow, поэтому CI не зависит от файла `mlflow.db`.

`mlflow.db`, `mlruns/` и `mlartifacts/` не коммитятся. Адрес хранилища в PowerShell: `$env:MLFLOW_TRACKING_URI="sqlite:///mlflow.db"`.

Порядок трёх запусков (@Z0rko32): для каждого запуска изменить `students/params.yaml`, затем `dvc repro students/dvc.yaml`, затем `train` и `evaluate` с `--mlflow`. После запусков вернуть `params.yaml`; `git status` должен быть чистым.

Сравнение конфигураций через `dvc metrics diff` (@GlooshSl): в main закоммитированы метрики выбранной по ADR конфигурации. В рабочей копии меняется `features.config`, выполняется `dvc repro students/dvc.yaml`, затем `dvc metrics diff --targets students/reports/metrics.json` сравнивает рабочую копию с последним коммитом. Результат вставляется в отчёт, рабочая копия возвращается в исходное состояние.

## Порядок слияния и адаптации

Правило: корневой конвейер ЛР3 и существующие тесты не меняются, поэтому CI остаётся зелёным на каждом PR. Новые файлы добавляются рядом.

| № | Что вливается | Кто | Зависит от | Что адаптируется |
|---|---|---|---|---|
| 1 | генератор и набор данных | @GlooshSl | - | добавляются новые файлы данных |
| 2 | `.gitignore` для MLflow | @kucheranton200 | - | исключения для `mlflow.db`, `mlruns/`, `mlartifacts/`, `students/reports/run.json` |
| 3 | extra `tracking` | @kerty0 | - | зависимости |
| 4 | CI с `.[dev,data,tracking]` | @kucheranton200 | №3 | установка в CI; без неё pytest не импортирует mlflow |
| 5 | контракт v2.0.0 и тесты | @kerty0 | №1 | `data/students_contract.json`; старый `contract.json` остаётся |
| 6 | preprocess, split, тесты | @Z0rko32 | №1, №5 | модули читают разделы `features` и `split` |
| 7 | ADR 0002 и ADR о конфигурации | @Vikabh | №1 | метрика и точка отсчёта фиксируются до запусков |
| 8 | train, evaluate, MLflow, тесты | @Z0rko32 | №4, №6 | `--mlflow`, skops, сигнатура, теги |
| 9 | `students/params.yaml`, `dvc.yaml`, `dvc.lock`, шаг CI | @kerty0 | №5, №6, №8 | в CI добавляется `dvc repro students/dvc.yaml`; перед PR `pytest` и `dvc repro` дважды |
| 10 | карточка данных | @Vikabh | №1 | новый раздел о наборе студентов |
| 11 | скрипт сравнения и бутстреп | @kucheranton200 | №8 | читает запуски эксперимента `student-dropout` |
| 12 | запуски эксперимента | @Z0rko32 | №7, №9, №11 | не PR; результат в локальном `mlflow.db` |
| 13 | отчёт `dvc metrics diff` | @GlooshSl | №9 | `docs/reports/feature-configs.md` |
| 14 | README: команды и таблица сравнения | @kerty0, @Vikabh | №11, №12 | разделы README независимы; таблица берётся из вывода скрипта сравнения |
| 15 | записка, карточка, отчёт, отклонения | @GlooshSl | №13, №14 | затем тег `lr4-v1` |
| 16 | evidence cards | каждый свою | №15 | затем тег `lr4-v1` |

ADR 0002 (№7) нужно влить до запусков (№12): метрика и точка отсчёта фиксируются до результата.

## График

| Когда | Что должно быть готово |
|---|---|
| Чт 29.10, до слияния плана | Ветка `lr4-base` (@kucheranton200) |
| Чт 29.10, вечер | План ЛР4 влит или на ревью, у каждого создана своя issue |
| Пт 30.10, 18:00 | PR с генератором и набором (@GlooshSl) |
| Пн 02.11, 12:00 | PR `.gitignore` (@kucheranton200), PR extra `tracking` (@kerty0) |
| Пн 02.11, 18:00 | PR CI (@kucheranton200), PR контракта (@kerty0) |
| Вт 03.11, 12:00 | PR preprocess и split (@Z0rko32), ADR 0002 и ADR о конфигурации (@Vikabh) |
| Вт 03.11, 20:00 | PR train, evaluate, MLflow (@Z0rko32) |
| Ср 04.11, 10:00 | PR `students/params.yaml` и `dvc.yaml` (@kerty0) |
| Ср 04.11, 12:00 | PR карточки данных (@Vikabh) |
| Ср 04.11, 14:00 | PR скрипта сравнения (@kucheranton200) |
| Ср 04.11, 16:00 | Три запуска выполнены (@Z0rko32) |
| Ср 04.11, 20:00 | Отчёт `dvc metrics diff` (@GlooshSl) |
| Чт 05.11, 10:00 | PR README: команды (@kerty0) и таблица сравнения (@Vikabh) |
| Чт 05.11, 12:00 | Повтор лучшего запуска на чистом клоне (@kucheranton200) |
| Чт 05.11, до 15:00 | Ревью всех PR, CI зелёный, всё влито |
| Чт 05.11, 16:00 | PR записки, карточки и отчёта (@GlooshSl) |
| Чт 05.11, 17:00 | PR evidence cards остальных участников; влиты до 18:00 |
| Чт 05.11, 18:00 | Tag `lr4-v1` и Release |
| Чт 05.11, 19:30 | Сводные PR (четверо, кроме @kucheranton200) |
| Чт 05.11, 20:00 | Ссылки в ДО.СКИФ |

## Сдача в ДО.СКИФ

Результат ЛР для не-тимлида: его PR в репозиторий тимлида со всеми его задачами за ЛР. Результат для тимлида: ссылка на main (по уточнению преподавателя в чате курса). Влитые PR объединить нельзя, поэтому каждый из четверых (@GlooshSl, @kerty0, @Z0rko32, @Vikabh) делает сводный PR, который не вливается.

1. Основание: ветка `lr4-base`, созданная до слияния плана.
2. Каждый после слияния всех PR (в том числе своей evidence card) выполняет в PowerShell:
   ```powershell
   git fetch origin
   git log origin/lr4-base..origin/main --no-merges --reverse --author="$(git config user.name)" --format="%h %an %s"
   ```
   Список должен содержать все свои коммиты. Если он пуст или неполный, имя в Git отличается от прежнего: повторить с `--author="$(git config user.email)"`.
3. Создать ветку и перенести коммиты по порядку (SHA из списка, от старых к новым):
   ```powershell
   git switch -c lr4-summary-<логин в нижнем регистре> origin/lr4-base
   git cherry-pick SHA1 SHA2 SHA3
   ```
   При конфликте: исправить файл, `git add .`, `git cherry-pick --continue`.
4. Выложить ветку: `git push -u origin lr4-summary-<логин в нижнем регистре>`.
5. Открыть PR по ссылке `https://github.com/kucheranton200/T26_MLOps/compare/lr4-base...lr4-summary-<логин в нижнем регистре>`. Слева должна быть ветка `lr4-base` репозитория kucheranton200/T26_MLOps. Кнопку Merge не нажимать.
6. В описании: что это сводный PR для сдачи, он не вливается, CI может быть красным (перенесены только свои коммиты, они опираются на чужие файлы); таблица «issue → PR в main → коммиты»; `Closes #N` не писать.
7. Если преподаватель потребует влитый PR со всеми коммитами, план пересматривается, это записывается в «Отклонения».

## Решения команды

Предложены координатором. Считаются принятыми после одобрения этого плана в PR.

1. **Фиксированные условия.** Одна версия данных (SHA-256 выше), одно стратифицированное разбиение (`test_size` 0,2, `random_state` 42), одинаковый расчёт метрик, одна модель: логистическая регрессия со стандартизацией и `class_weight=balanced` внутри одного `Pipeline`. Меняется только набор признаков.
2. **Масштабирование внутри модели.** Скейлер обучается только на train и сохраняется вместе с моделью. Отдельный файл скейлера не создаётся: pickle в проекте запрещён.
3. **Сериализация.** Модель сохраняется в формате skops. У логистической регрессии список недоверенных типов пуст; переход на pickle для обхода проверки отклонён и записывается в ADR 0002.
4. **Оценка разброса.** Логистическая регрессия детерминирована, поэтому по seed разброс был бы нулевым. Используем бутстреп по тесту (1000 повторов с возвращением): 95 %-интервал macro-F1 и F1 Dropout для каждого запуска и парный бутстреп разности «поздняя минус ранняя». Различие меньше ширины интервала улучшением не считается; итоговую формулировку правила фиксирует ADR 0002 до запусков.
5. **Конфигурация в main.** Решение фиксирует ADR о конфигурации признаков. Предложение: по умолчанию в main ранняя конфигурация, так как поздняя недоступна куратору в момент зачисления. Поздняя остаётся как сравниваемый вариант.
6. **Честность выбора.** Выбор делается среди трёх запусков, число попыток записывается в README. Если разница меньше разброса, это пишется прямо; отрицательный результат тоже результат.
7. **Reviewer.** Лучший запуск повторяет @kucheranton200 на чистом клоне с собственным `mlflow.db`; метрика должна совпасть.
8. **Артефакты коллег.** Каждый пишет свои файлы и свою evidence card, в ней честно указано использование ИИ.
9. **Независимость от ЛР3.** Вариант 10 строится отдельным конвейером в `students/`. Файлы и тег ЛР3 не меняются и не удаляются.

## Порядок работы

1. У каждого участника своя issue с наблюдаемым результатом и критериями приёмки.
2. Работа идёт в ветках `lab4/<название>` от свежего main. В main только через PR, в описании `Closes #N`.
3. PR открывается по ссылке `https://github.com/kucheranton200/T26_MLOps/compare/main...ИМЯ_ВЕТКИ`; слева должно быть `kucheranton200/T26_MLOps`, а не репозиторий преподавателя.
4. Merge после зелёного CI (Ruff, pytest, `dvc repro`) и одобрения другого участника.
5. Tag `lr4-v1` ставится после влития всех PR и не переставляется.
6. Сдача в ДО.СКИФ: раздел «Сдача в ДО.СКИФ».

## Зависимости и риски

- **Репозиторий сдачи.** Страница сдачи ссылается на Irina-64/T25--4m_MLOps. Уточнить у преподавателя, куда открывать PR.
- **Формат сдачи.** Сводный PR без влития (раздел «Сдача в ДО.СКИФ») принят командой, но преподавателем пока не подтверждён.
- **Пути в `dvc.yaml`.** Выходы лежат в `students/`, команды запускаются из корня. Способ задать пути (`wdir: ..` или `../` в путях) выбирает @kerty0 и проверяет через `dvc repro` дважды и `dvc params diff`; выбор записывается в описании PR.
- **Валидатор.** Существующий `validate_data` может не поддерживать нужные проверки для контракта студентов; запасной путь описан в разделе «Границы работы».
- **Skops в MLflow.** Способ сохранения модели в формате skops с сигнатурой проверяет @Z0rko32; если `log_model` его не поддерживает, `model.skops` логируется артефактом, а сигнатура и входной пример записываются отдельно. Выбор записывается в описании PR.
- **Утечка и запрещённые признаки.** Target не входит в признаки; преобразования обучаются только на train.
- **Порядок PR.** Стадия train нужна до `dvc.yaml`: иначе `dvc repro` нечем запускать.
- **CRLF.** Файлы приводятся к LF до `dvc repro`; проверка `git ls-files --eol` обязательна, иначе в чистом клоне `dvc status` покажет ложные изменения.
- **Служебные файлы.** `mlflow.db`, `mlruns/`, `mlartifacts/` не коммитятся.
- **Конфликты README.** Два PR правят README одновременно; разделы разные, второй PR перед слиянием обновляется от main.
- **Нет ответа участника до срока.** Задача переходит GlooshSl или @kucheranton200 по договорённости.

## Отклонения от плана

Сюда записываются передача и изменение задач по договорённости в ходе работы. Строки основной таблицы не переписываются: так видно, что было запланировано и что изменилось.

| Дата | Задача | Было | Стало | Причина | Кто согласовал |
|---|---|---|---|---|---|
| | | | | | |
