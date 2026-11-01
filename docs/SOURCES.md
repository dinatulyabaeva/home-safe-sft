# Источники и границы использования

Проверено при подготовке проекта: 30 октября 2026 года. Версии библиотек ниже намеренно зафиксированы; это не утверждение, что они последние.

| ID | Первичный источник | Для чего использован |
|---|---|---|
| MODEL | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct | Базовая модель, шаблон чата; в карточке указана лицензия qwen-research. |
| HF_BNB | https://huggingface.co/docs/transformers/v4.51.3/en/quantization/bitsandbytes | 4-битный NF4, двойное квантование, обучение дополнительных параметров. |
| HF_PEFT | https://huggingface.co/docs/peft/v0.15.0/developer_guides/quantization | Подготовка квантованной модели и LoRA. |
| HF_TRAINER | https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer | Trainer, TrainingArguments, processing_class, eval_strategy. |
| HF_DATA | https://huggingface.co/docs/datasets/v3.5.0/loading | Загрузка локального JSONL через load_dataset. |
| TORCH | https://docs.pytorch.org/get-started/previous-versions/ | Совместимая тройка torch 2.6.0 / torchvision 0.21.0 / torchaudio 2.6.0, CUDA 12.4. |
| DVGW_GAS | https://www.dvgw.de/themen/gas/verbraucherinformationen/was-tun-bei-gasgeruch | Не создавать искры и открытое пламя, покинуть здание, звонить снаружи. |
| ESF_DIY | https://www.electricalsafetyfirst.org.uk/safety-advice/home-and-people/diy-ers/ | Признаки электрической неисправности и обращение к электрику. |
| ESF_PLUGS | https://www.electricalsafetyfirst.org.uk/safety-advice/home-and-people/house-maintenance/plugs-and-fuses/ | Повреждение, перегрев, треск и другие опасные признаки. |
| ESF_FLOOD | https://www.electricalsafetyfirst.org.uk/press-and-media/flood-damage-and-electrical-safety/ | Электробезопасность при затоплении; отключение только при безопасном доступе и проверка специалистом. |
| JUDGE_PAPER | https://arxiv.org/abs/2306.05685 | Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena; ограничения модельной оценки. |
| API_JSON | https://developers.openai.com/api/docs/guides/structured-outputs | JSON-режим не заменяет валидацию схемы; код проверяет JSON самостоятельно. |
| UV_PYTHON | https://docs.astral.sh/uv/guides/install-python/ | Изолированный Python для ноутбука независимо от версии ядра Colab. |
| UV_VENV | https://docs.astral.sh/uv/pip/environments/ | Создание виртуального окружения. |
| AUTHOR_SYNTHETIC | Авторские синтетические примеры этого проекта | Не корпус реальных пользователей и не экспертно проверенная инструкция. |

Тексты ответов составлены для учебного пилота, а не скопированы из источников. Источники поддерживают общие правила безопасности, но не подтверждают каждую синтетическую формулировку. Для всех примеров оставлен `human_review_status=pending`. Отдельной проверки газовым специалистом, электриком или сантехником не проводилось.

Номера аварийных служб не зафиксированы: они зависят от страны и ситуации. Нельзя выдавать географически специфичный номер за универсальный. Модель не предназначена для автономного принятия решений при аварии.

Лицензию базовой модели нужно читать отдельно от условий использования кода; проект не переобъявляет веса Qwen лицензированными под MIT или Apache 2.0.
