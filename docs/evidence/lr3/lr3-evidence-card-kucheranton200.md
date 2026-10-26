# Evidence Card ЛР3 — Кучер Антон (@kucheranton200)

**Роль в ЛР3:** Repository administrator  
**Команда:** team-2  
**Репозиторий:** https://github.com/kucheranton200/T26_MLOps  
**Ветки:** `feature/dvc-init`, `lab3/ci-dvc-repro`  
**Issues:** [#47](https://github.com/kucheranton200/T26_MLOps/issues/47), [#53](https://github.com/kucheranton200/T26_MLOps/issues/53)  
**Pull Requests:** [#48](https://github.com/kucheranton200/T26_MLOps/pull/48), [#54](https://github.com/kucheranton200/T26_MLOps/pull/54)  
**Тег:** `lr3-v1` _(ставится координатором @GlooshSl после слияния всех PR)_

---

## 1. Что сделано

| Артефакт / Задача | Путь | Назначение |
|---|---|---|
| Инициализация DVC | `.dvc/`, `.dvc/config`, `.dvc/.gitignore` | Базовая конфигурация DVC в репозитории |
| Файл исключений DVC | `.dvcignore` | Исключение временных файлов и кэшей (`.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`) |
| Настройка исключений Git | `.gitignore` | Добавление исключений для внутренних файлов DVC (`.dvc/cache/`, `.dvc/tmp/`, `.dvc/config.local`) и защита от попадания промежуточных данных в репозиторий |
| Стратегия хранения данных | `data/flights_sample.csv` | Оставлен в Git и указан зависимостью в `dvc.yaml` без перевода под `dvc add` — обеспечивает запуск конвейера на чистом клоне и в CI без DVC remote |
| Шаг конвейера в CI | `.github/workflows/ci.yml` | Добавление шага `dvc repro` в CI workflow перед запуском тестов |
| Актуализация плана ЛР3 | `docs/plans/lr3-plan.md` | Привязка созданного issue [#53](https://github.com/kucheranton200/T26_MLOps/issues/53) к задаче шага `dvc repro` в CI |
| Личная evidence-card | `docs/evidence/lr3/lr3-evidence-card-kucheranton200.md` | Подтверждение выполнения роли Repository administrator, команд, решений и анализа рисков |

---

## 2. Команды запуска и проверки

```bash
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

## 3. Интеграция шага `dvc repro` в CI (`.github/workflows/ci.yml`)

### Конфигурация шага

```yaml
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.12"
          cache: pip
      - run: python -m pip install --upgrade pip
      - run: python -m pip install -e ".[dev]"
      - run: python -m ruff check .
      - run: python -m mlops_project.validate_data --data data/flights_sample.csv --contract data/contract.json --report reports/validation/lr2.json
      - run: dvc repro
      - run: python -m pytest -q
```

### Обоснование порядка выполнения

1. `pip install -e ".[dev]"` устанавливает пакет и все зависимости, включая `dvc>=3.0.0`, регистрируя CLI-утилиту `dvc` в виртуальном окружении.
2. `ruff check .` и валидатор данных `mlops_project.validate_data` выполняются первыми как входной шлюз качества кода и контракта входных данных (требование ЛР2 сохранено).
3. `dvc repro` запускает весь граф стадий (`validate` → `preprocess` → `split` → `train` → `evaluate`). В чистом окружении runner'а GitHub Actions генерируются артефакты `data/processed/`, `data/split/`, `models/`, `reports/metrics.json`.
4. `pytest -q` выполняется **после** `dvc repro`, благодаря чему тесты конвейера (`tests/test_pipeline.py`) могут валидировать как независимые модули, так и фактически сформированные артефакты на диске.

### Подтверждение успешного прогона в CI

В рамках [Pull Request #54](https://github.com/kucheranton200/T26_MLOps/pull/54) обе проверки GitHub Actions завершились со статусом **PASS** (зелёный CI):
* `CI/quality (push)` — Passed (~42s)
* `CI/quality (pull_request)` — Passed (~40s)

---

## 4. Воспроизводимость на чистом клоне

Конвейер спроектирован так, чтобы работать "из коробки" на чистом клоне репозитория без предварительной настройки удалённых хранилищ (S3, GCS, SSH):

```bash
# Имитация работы на чистом клоне:
git clone https://github.com/kucheranton200/T26_MLOps.git
cd T26_MLOps
python -m venv .venv
source .venv/bin/activate  # или .\.venv\Scripts\Activate.ps1 на Windows
pip install -e ".[dev]"

# Запуск конвейера с нуля:
dvc repro
```

**Поведение DVC на чистом клоне:**
* Так как в репозитории отсутствуют промежуточные кэшированные файлы (`data/processed/`, `data/split/`, `models/*.pkl`), DVC автоматически исполняет стадии конвейера на основе `data/flights_sample.csv` и параметров `params.yaml`.
* Все артефакты генерируются детерминированно (фиксированный seed=42 в `params.yaml`).
* При повторном запуске `dvc repro` DVC сообщает: `Data and pipelines are up to date.` — все 5 стадий пропускаются.

---

## 5. Стратегия управления данными (Data Management Decision)

### Решение по `data/flights_sample.csv`

1. **Размер и безопасность:** Файл `data/flights_sample.csv` представляет собой синтетическую выборку из 24 строк размером ~214 байт, не содержит персональных или коммерческих данных.
2. **Хранение в Git:** Файл оставлен под версионным контролем Git и не переводится под `dvc add`.
3. **Роль в DVC:** В `dvc.yaml` путь `data/flights_sample.csv` указан в секции `deps` стадий `validate` и `preprocess`. DVC отслеживает его целостность по хешу SHA-256.
4. **Преимущество:** Полная автономность CI и разработчиков — не требуется настраивать и оплачивать cloud storage или передавать секреты/ключи доступа через GitHub Actions Secrets.

---

## 6. Проверка правил игнорирования (`git check-ignore`)

Команда `git check-ignore -v` подтверждает корректность разграничения:

```bash
# Выходные файлы и модели игнорируются Git:
$ git check-ignore -v data/processed/flights_processed.csv data/split/X_train.csv models/model.pkl models/scaler.pkl
.gitignore:19:data/processed/           data/processed/flights_processed.csv
data/split/.gitignore:1:/X_train.csv    data/split/X_train.csv
.gitignore:20:models/                   models/model.pkl
.gitignore:20:models/                   models/scaler.pkl

# Исходные файлы, параметры, пайплайн и lock-файл НЕ игнорируются (вывод пуст):
$ git check-ignore -v data/flights_sample.csv data/contract.json params.yaml dvc.yaml dvc.lock
(пустой вывод)
```

В Git коммитятся только управляющие файлы конвейера (`dvc.yaml`, `dvc.lock`, `params.yaml`) и отчёты с `cache: false` (`reports/metrics.json`, `reports/validation/lr3.json`).

---

## 7. Анализ рисков роли Repository administrator

### Риск 1: Зависимость от незаданного DVC remote в CI
* **Описание:** Если бы данные отслеживались только через `dvc add`, на чистом runner'е GitHub Actions команда `dvc repro` упала бы из-за отсутствия файлов данных и невозможности сделать `dvc pull` без настроенного remote и credentials.
* **Митигация:** Исходный файл зафиксирован в Git и отслеживается как `deps` в `dvc.yaml`. Все промежуточные данные генерируются конвейером на лету.

### Риск 2: Случайное выполнение `dvc install`
* **Описание:** Команда `dvc install` устанавливает Git hooks (включая `pre-push` с вызовом `dvc push`). При отсутствии настроенного remote любой `git push` завершался бы фатальной ошибкой.
* **Митигация:** Зафиксировано соглашение в плане ЛР3 — не вызывать `dvc install` до появления удалённого хранилища.

### Риск 3: Рассинхронизация окончаний строк (CRLF vs LF) и хешей DVC
* **Описание:** На машинах Windows при checkout без нормализации EOL файлы могут получить `CRLF`, что изменяет размер и MD5-хеш текстовых файлов (JSON, CSV, YAML), приводя к ложным срабатываниям `dvc repro` и конфликтам в `dvc.lock`.
* **Митигация:** В репозитории зафиксирован файл `.gitattributes` с правилом `* text=auto eol=lf`, что гарантирует единообразие хэшей на Linux, macOS и Windows.

### Риск 4: Разрастание репозитория за счёт тяжёлых артефактов
* **Описание:** Попадание артефактов стадий (бинарные модели `.pkl`, промежуточные датасеты) в Git раздувает историю репозитория.
* **Митигация:** Многоуровневая защита через корневой `.gitignore` (`data/processed/`, `models/`, `*.pkl`) и локальный `data/split/.gitignore`.

---

## 8. Итоги и чек-лист готовности

- [x] Инициализирован DVC, настроены `.dvcignore` и `.gitignore` (Issue #47, PR #48)
- [x] Зафиксировано архитектурное решение по данным без внешнего storage
- [x] В `.github/workflows/ci.yml` внедрён шаг `dvc repro` (Issue #53, PR #54)
- [x] CI в GitHub Actions зелёный для всех триггеров
- [x] `ruff check .` и `pytest -q` проходят успешно
- [x] Создана персональная evidence-card

---

_Evidence card заполнена: 2026-10-26. Контрольный тег `lr3-v1` и итоговый SHA будут проставлены координатором @GlooshSl после завершения этапа слияния всех PR ЛР3._
