import json
import math
import re
from datetime import datetime
from pathlib import Path

import jsonschema

from src.cv import CV_SCHEME, LEGACY_CV_SCHEMES, N_SPLITS

RESULTS_FILE = "results.json"

# 人間の起案は exp001、Claude Code の起案は expA001。アイデアの出所を ID だけで区別するため
ORIGIN_PATTERNS = {"human": r"^exp\d{3}$", "claude": r"^expA\d{3}$"}

# 丸めて手入力された値も許すため、完全一致ではなく許容誤差で比べる
CV_TOLERANCE = 1e-4

_NULLABLE_NUMBER = {"type": ["number", "null"]}
_NON_EMPTY = {"type": "string", "minLength": 1}

RESULTS_SCHEMA = {
    "type": "object",
    "required": [
        "exp_id",
        "origin",
        "parent",
        "created_at",
        "git_hash",
        "hypothesis",
        "changes",
        "model",
        "config",
        "metric",
        "cv",
        "lb",
        "train_time_sec",
        "status",
        "conclusion",
        "tags",
    ],
    # フィールド名の打ち間違いを見逃さないため、未定義のキーを拒否する
    "additionalProperties": False,
    "properties": {
        "exp_id": {"type": "string", "pattern": r"^expA?\d{3}$"},
        "origin": {"enum": list(ORIGIN_PATTERNS)},
        "parent": {"type": ["string", "null"]},
        "created_at": _NON_EMPTY,
        "git_hash": {"type": "string", "pattern": r"^[0-9a-f]{7,40}(-dirty)?$"},
        "hypothesis": _NON_EMPTY,
        "changes": _NON_EMPTY,
        "model": _NON_EMPTY,
        "config": {"type": "object"},
        "metric": {
            "type": "object",
            "required": ["name", "direction"],
            "additionalProperties": False,
            "properties": {"name": _NON_EMPTY, "direction": {"enum": ["minimize", "maximize"]}},
        },
        "cv": {
            "type": ["object", "null"],
            "required": ["scheme", "mean", "std", "folds"],
            "additionalProperties": False,
            "properties": {
                "scheme": {"enum": [CV_SCHEME, *LEGACY_CV_SCHEMES]},
                "mean": {"type": "number"},
                "std": {"type": "number", "minimum": 0},
                "folds": {"type": "array", "items": {"type": "number"}, "minItems": 1},
            },
        },
        "lb": {
            "type": "object",
            "required": ["public", "private"],
            "additionalProperties": False,
            "properties": {"public": _NULLABLE_NUMBER, "private": _NULLABLE_NUMBER},
        },
        "train_time_sec": {"type": ["number", "null"], "minimum": 0},
        "status": {"enum": ["running", "done", "failed"]},
        "conclusion": {"type": "string"},
        "tags": {"type": "array", "items": _NON_EMPTY},
        "wandb_url": {"type": "string", "pattern": r"^https://"},
    },
}


def validate(results, exp_dir, known_ids):
    """スキーマに加えて、スキーマでは表せないフィールド間の整合も検査する。"""
    errors = [
        f"{'.'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}"
        for e in jsonschema.Draft202012Validator(RESULTS_SCHEMA).iter_errors(results)
    ]
    if errors:
        return errors, []

    warnings = []
    exp_id = results["exp_id"]
    if exp_id != exp_dir.name:
        errors.append(f"exp_id {exp_id} がディレクトリ名 {exp_dir.name} と一致しない")
    if not re.match(ORIGIN_PATTERNS[results["origin"]], exp_id):
        errors.append(f"origin {results['origin']} の実験は {ORIGIN_PATTERNS[results['origin']]} で命名する")

    parent = results["parent"]
    if parent == exp_id:
        errors.append("parent に自分自身を指定している")
    elif parent is not None and parent not in known_ids:
        errors.append(f"parent {parent} が experiments/ に存在しない")

    try:
        created_at = datetime.fromisoformat(results["created_at"])
    except ValueError:
        errors.append(f"created_at {results['created_at']} が ISO 8601 形式でない")
    else:
        if created_at.tzinfo is None:
            errors.append("created_at にタイムゾーンがない")

    cv = results["cv"]
    if results["status"] == "done":
        if cv is None:
            errors.append("status が done なのに cv が null")
        if results["train_time_sec"] is None:
            errors.append("status が done なのに train_time_sec が null")
        if not results["conclusion"]:
            warnings.append("conclusion が未記入")
    if cv is not None:
        folds = cv["folds"]
        if len(folds) != N_SPLITS:
            errors.append(f"cv.folds が {len(folds)} 件。src/cv.py の N_SPLITS={N_SPLITS} と一致しない")
        mean = sum(folds) / len(folds)
        std = math.sqrt(sum((f - mean) ** 2 for f in folds) / len(folds))
        if not math.isclose(cv["mean"], mean, abs_tol=CV_TOLERANCE):
            errors.append(f"cv.mean {cv['mean']} が folds の平均 {mean:.6f} と一致しない")
        if not math.isclose(cv["std"], std, abs_tol=CV_TOLERANCE):
            errors.append(f"cv.std {cv['std']} が folds の標準偏差 {std:.6f} と一致しない")

    return errors, warnings


def load_all(experiments_dir):
    """experiments/*/results.json を読み、実験ごとに (results, errors, warnings) を返す。"""
    paths = sorted(Path(experiments_dir).glob(f"*/{RESULTS_FILE}"))
    known_ids = {p.name for p in Path(experiments_dir).iterdir() if p.is_dir()}
    report = []
    for path in paths:
        try:
            results = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            report.append((path, None, [f"JSON として読めない: {e}"], []))
            continue
        errors, warnings = validate(results, path.parent, known_ids)
        report.append((path, results, errors, warnings))
    return report
