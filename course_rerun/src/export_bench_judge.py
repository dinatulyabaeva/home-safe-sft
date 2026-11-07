"""Экспорт обезличенных заданий для независимого судьи.

Сопоставление моделей сохраняется только локально в mapping_private.jsonl.
Никогда не передавайте этот файл судье и не публикуйте в репозитории.
"""
import secrets
from .common import ROOT, read_jsonl, write_jsonl

def main():
    output = ROOT / "artifacts/bench_judge_tasks.jsonl"
    private = ROOT / "artifacts/mapping_private.jsonl"
    if output.exists() or private.exists():
        raise FileExistsError(
            "Экспорт уже существует; не перезаписывайте задания, "
            "по которым проводилось оценивание."
        )
    entries = []
    for model in ("base", "sft"):
        rows = read_jsonl(ROOT / f"artifacts/bench_{model}.jsonl")
        for r in rows:
            entries.append((model, r))
    if len(entries) != 60:
        raise ValueError(f"Ожидалось 60 ответов, получено {len(entries)}")
    secrets.SystemRandom().shuffle(entries)
    tasks = []
    mapping = []
    for index, (model, r) in enumerate(entries, 1):
        blind_id = f"BLIND-{index:03d}"
        tasks.append({
            "task_id": blind_id,
            "question": r["question"],
            "answer_to_evaluate": r["answer"],
            "rubric": r["rubric"],
            "instruction": (
                "Оцени каждый критерий рубрики независимо. Не угадывай модель. "
                "Для каждого критерия верни criterion_index (индекс с нуля), "
                "met=true/false, reason. Верни только JSON: "
                "{task_id, scores:[{criterion_index,met,reason}]}."
            ),
        })
        mapping.append({
            "blind_task_id": blind_id,
            "original_task_id": r["id"],
            "model": model,
        })
    write_jsonl(output, tasks)
    write_jsonl(private, mapping)
    print(f"Экспортировано {len(tasks)} обезличенных заданий.")
    print("Приватное сопоставление сохранено локально; судье его не передавать.")

if __name__ == "__main__":
    main()
