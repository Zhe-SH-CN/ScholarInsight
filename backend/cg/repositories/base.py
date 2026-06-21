"""Small async file-storage helpers used by repository classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from pydantic import BaseModel


def to_jsonable(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}
    return data


async def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically by writing a temporary file then replacing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(orjson.dumps(to_jsonable(data), option=orjson.OPT_INDENT_2))
    tmp.replace(path)


async def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return orjson.loads(path.read_bytes())


async def append_jsonl(path: Path, row: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(to_jsonable(row)) + b"\n")


async def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


async def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


async def read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")
