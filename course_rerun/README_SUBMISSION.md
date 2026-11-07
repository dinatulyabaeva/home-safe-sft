# Сдача рубежного контроля № 1 — повторный эксперимент

## Уже опубликовано

- [Итоговый Markdown-отчёт с формулами](reports/FINAL_REPORT.md).
- [Официальные данные курса](data/), [код](src/), [ноутбук](notebooks/home_course_sft_colab.ipynb), [конфигурация](configs/experiment.json).
- [LoRA-адаптер](artifacts/course_adapter/) (включая `adapter_model.safetensors`).
- [30 ответов Baseline](artifacts/bench_base.jsonl), [30 ответов SFT](artifacts/bench_sft.jsonl).
- [60 слепых заданий](artifacts/bench_judge_tasks.jsonl) и [60 результатов независимого судьи](artifacts/bench_judge_results_rerun.jsonl).
- [Рубричная сводка](artifacts/rubric_comparison_rerun.json), [проверка формата](artifacts/format_metrics_rerun.json), [парное сравнение](artifacts/paired_comparison_rerun.csv), [журнал обучения](artifacts/course_training_metrics.json).

## Ещё нужно

1. **Загрузить итоговый Word-отчёт** `otchet_qlora_home_rk1_final.docx` в папку `course_rerun/reports/`; при требовании LMS экспортировать PDF.
2. При необходимости получить validation-генерации и рассчитать `urgency_accuracy`, `who_accuracy`, `category_accuracy` и `difficulty_mae` через `python -m src.generate --split validation --model base`, затем `--model sft`, затем `python -m src.evaluate`. Эти файлы сейчас отсутствуют.
3. Убедиться, что все ссылки открываются, и отправить преподавателю ссылку именно на `course_rerun/`, а не на старый корневой отчёт.

**Безопасность:** токены, `.env` и `mapping_private.jsonl` не публиковать. Судье передавать только обезличенные задания. Старые материалы в корне репозитория относятся к другому эксперименту на 192 синтетических примерах.
