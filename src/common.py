from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{n}: invalid JSON: {e}') from e
            if not isinstance(obj, dict):
                raise ValueError(f'{path}:{n}: expected a JSON object')
            rows.append(obj)
    return rows

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')

def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def unique_index(rows: list[dict], key: str = 'id') -> dict:
    result = {}
    for r in rows:
        if key not in r or r[key] in result:
            raise ValueError(f'Missing or duplicate {key}: {r.get(key)}')
        result[r[key]] = r
    return result
