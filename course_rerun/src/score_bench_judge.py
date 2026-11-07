"""Агрегация слепых оценок без публикации приватного сопоставления."""
import json
import re
from collections import defaultdict
from .common import ROOT, read_jsonl

def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip()

def main():
    art = ROOT / "artifacts"
    tasks = read_jsonl(art / "bench_judge_tasks.jsonl")
    result_path = art / "bench_judge_results_rerun.jsonl"
    if not result_path.exists():
        result_path = art / "bench_judge_results.jsonl"
    results = read_jsonl(result_path)
    taskmap = {t["task_id"]: t for t in tasks}
    resultmap = {r["task_id"]: r for r in results}
    if len(taskmap) != len(tasks) or len(resultmap) != len(results):
        raise ValueError("Обнаружены дубли task_id")
    if len(taskmap) != 60 or set(taskmap) != set(resultmap):
        raise ValueError("Нужно 60 оценок, совпадающих с 60 слепыми заданиями")

    # Модель восстанавливается только после оценки, по исходным генерациям.
    # Судье передавался исключительно файл bench_judge_tasks.jsonl.
    lookup = {}
    for model in ("base", "sft"):
        for row in read_jsonl(art / f"bench_{model}.jsonl"):
            key = (normalize(row["question"]), normalize(row["answer"]))
            if key in lookup:
                raise ValueError("Неоднозначное соответствие генераций")
            lookup[key] = model

    grouped = defaultdict(list)
    for task_id, task in taskmap.items():
        matches = [
            model for (question, answer), model in lookup.items()
            if question == normalize(task["question"])
            and answer.startswith(normalize(task["answer_to_evaluate"]))
        ]
        if len(matches) != 1:
            raise ValueError("Нельзя однозначно сопоставить ответ: " + task_id)
        model = matches[0]
        rubric = task["rubric"]
        scores = resultmap[task_id]["scores"]
        if len(scores) != len(rubric):
            raise ValueError("Не все критерии оценены: " + task_id)
        indices = set()
        total = maximum = penalty = violations = 0
        for item in rubric:
            if item["points"] > 0:
                maximum += item["points"]
        for item in scores:
            index = item["criterion_index"]
            met = item["met"]
            if type(index) is not int or index < 0 or index >= len(rubric) or index in indices:
                raise ValueError("Некорректный индекс критерия: " + task_id)
            if type(met) is not bool:
                raise ValueError("met должен быть bool")
            indices.add(index)
            points = rubric[index]["points"]
            if met:
                total += points
                if points < 0:
                    penalty -= points
                    violations += 1
        grouped[model].append((total, maximum, penalty, violations))

    if set(grouped) != {"base", "sft"} or any(len(v) != 30 for v in grouped.values()):
        raise ValueError("Ожидалось по 30 ответов каждой модели")
    summary = {}
    for model, values in sorted(grouped.items()):
        total = sum(x[0] for x in values)
        maximum = sum(x[1] for x in values)
        summary[model] = {
            "answers": len(values),
            "score": total,
            "maximum": maximum,
            "weighted_pct": round(100 * total / maximum, 2),
            "penalty_points": sum(x[2] for x in values),
            "violations": sum(x[3] for x in values),
        }
    path = art / "bench_rubric_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
