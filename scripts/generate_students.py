"""Скрипт генерации синтетического набора данных «Отчисление студентов» (вариант 10).

Генерирует воспроизводимый CSV-файл на 1200 строк с целочисленными признаками
для ранней и поздней конфигураций оценки риска отчисления студентов.
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path


def generate_students(seed: int = 42, n_samples: int = 1200) -> list[dict[str, int]]:
    """Генерирует список словарей со строками датасета студентов.

    Распределение целевого класса Target:
    - 0 (Dropout):  341 строка (~28.4%)
    - 1 (Enrolled): 171 строка (~14.3%)
    - 2 (Graduate): 688 строк (~57.3%)
    Точка отсчёта macro-F1 мажоритарного baseline: ≈ 0.243.
    """
    rng = random.Random(seed)

    n_dropout = 341
    n_enrolled = 171
    n_graduate = 688
    assert n_dropout + n_enrolled + n_graduate == n_samples

    rows: list[dict[str, int]] = []

    for target, count in [(0, n_dropout), (1, n_enrolled), (2, n_graduate)]:
        for _ in range(count):
            if target == 0:  # Dropout
                age = rng.randint(19, 45)
                gender = 1 if rng.random() < 0.65 else 0
                daytime = 1 if rng.random() < 0.70 else 0
                scholarship = 1 if rng.random() < 0.08 else 0
                prev_grade = rng.randint(100, 145)
                adm_grade = rng.randint(100, 145)

                sem1_enrolled = rng.randint(5, 7)
                sem1_approved = rng.randint(0, min(2, sem1_enrolled))
                sem1_grade = rng.randint(10, 12) if sem1_approved > 0 else 0

                sem2_enrolled = rng.randint(5, 7)
                sem2_approved = rng.randint(0, min(1, sem2_enrolled))
                sem2_grade = rng.randint(10, 12) if sem2_approved > 0 else 0

            elif target == 1:  # Enrolled
                age = rng.randint(18, 30)
                gender = 1 if rng.random() < 0.50 else 0
                daytime = 1 if rng.random() < 0.85 else 0
                scholarship = 1 if rng.random() < 0.20 else 0
                prev_grade = rng.randint(115, 160)
                adm_grade = rng.randint(115, 160)

                sem1_enrolled = rng.randint(5, 7)
                sem1_approved = rng.randint(2, min(5, sem1_enrolled))
                sem1_grade = rng.randint(11, 14) if sem1_approved > 0 else 0

                sem2_enrolled = rng.randint(5, 7)
                sem2_approved = rng.randint(2, min(5, sem2_enrolled))
                sem2_grade = rng.randint(11, 14) if sem2_approved > 0 else 0

            else:  # Graduate
                age = rng.randint(17, 24)
                gender = 1 if rng.random() < 0.40 else 0
                daytime = 1 if rng.random() < 0.95 else 0
                scholarship = 1 if rng.random() < 0.42 else 0
                prev_grade = rng.randint(125, 185)
                adm_grade = rng.randint(125, 185)

                sem1_enrolled = rng.randint(5, 7)
                sem1_approved = rng.randint(4, sem1_enrolled)
                sem1_grade = rng.randint(13, 18)

                sem2_enrolled = rng.randint(5, 7)
                sem2_approved = rng.randint(4, sem2_enrolled)
                sem2_grade = rng.randint(13, 18)

            row = {
                "Age": age,
                "Gender": gender,
                "DaytimeAttendance": daytime,
                "Scholarship": scholarship,
                "PrevGrade": prev_grade,
                "AdmissionGrade": adm_grade,
                "Sem1Enrolled": sem1_enrolled,
                "Sem1Approved": sem1_approved,
                "Sem1Grade": sem1_grade,
                "Sem2Enrolled": sem2_enrolled,
                "Sem2Approved": sem2_approved,
                "Sem2Grade": sem2_grade,
                "Target": target,
            }
            rows.append(row)

    rng.shuffle(rows)
    return rows


def save_dataset(rows: list[dict[str, int]], output_path: Path) -> tuple[str, int]:
    """Сохраняет строки в CSV-файл строго с LF и вычисляет SHA-256."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Age",
        "Gender",
        "DaytimeAttendance",
        "Scholarship",
        "PrevGrade",
        "AdmissionGrade",
        "Sem1Enrolled",
        "Sem1Approved",
        "Sem1Grade",
        "Sem2Enrolled",
        "Sem2Approved",
        "Sem2Grade",
        "Target",
    ]

    with open(output_path, "w", newline="\n", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(output_path, "rb") as f:
        data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()
        size = len(data)

    return sha256, size


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор синтетического датасета студентов.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/students_sample.csv"),
        help="Путь к результирующему CSV файлу.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed генератора случайных чисел для воспроизводимости.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Количество строк данных.",
    )
    args = parser.parse_args()

    rows = generate_students(seed=args.seed, n_samples=args.samples)
    sha256, size = save_dataset(rows, args.output)

    print(f"Датасет успешно сгенерирован: {args.output}")
    print(f"Строк: {len(rows)}")
    print(f"Размер: {size} байт")
    print(f"SHA-256: {sha256}")


if __name__ == "__main__":
    main()
