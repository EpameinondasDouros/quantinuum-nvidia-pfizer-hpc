"""Local experiment persistence for the workbench (Lane E).

    store = ExperimentStore()                       # runs/experiments/ (gitignored)
    eid = store.create(config, "transformer_sft", record=record)
    store.add_circuit(eid, name, circuit); store.save_grades(eid, {name: report})
    store.add_execution(eid, execution); store.save_export(eid, "circuit.qasm", text)
    store.list(); store.load(eid); store.delete(eid)

Layout of runs/experiments/<experiment_id>/: config.json, record.json, circuits.jsonl, grades.json,
executions.jsonl, exports/. An experiment id is "<utc timestamp>-<method>-<config hash>"; ids and export
file names are validated against strict patterns and never joined from free user text.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path

from genqai.export import sha256_of, to_jsonable
from genqai.schema import GeneratedCircuit, HamiltonianRecord

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "runs" / "experiments"
ID_RE = re.compile(r"^\d{8}T\d{12}Z-[a-z0-9_]{1,40}-[0-9a-f]{12}$")
EXPORT_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}$")


def _method_slug(method: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", method.lower()).strip("_")[:40]
    return slug or "method"


def experiment_id(config, method: str, now: dt.datetime | None = None) -> str:
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    return f"{now:%Y%m%dT%H%M%S%f}Z-{_method_slug(method)}-{sha256_of(config)[:12]}"


class ExperimentStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)

    def _dir(self, eid: str, must_exist: bool = True) -> Path:
        if not isinstance(eid, str) or not ID_RE.match(eid):
            raise ValueError(f"invalid experiment id {eid!r}")
        d = self.root / eid
        if must_exist and not d.is_dir():
            raise KeyError(f"no experiment {eid}")
        return d

    @staticmethod
    def _write(path: Path, obj) -> None:
        path.write_text(json.dumps(to_jsonable(obj), indent=2))

    @staticmethod
    def _append(path: Path, obj) -> None:
        with path.open("a") as f:
            f.write(json.dumps(to_jsonable(obj)) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []

    def create(self, config, method: str, *, record: HamiltonianRecord | None = None, meta: dict | None = None,
               now: dt.datetime | None = None) -> str:
        eid = experiment_id(config, method, now)
        d = self._dir(eid, must_exist=False)
        (d / "exports").mkdir(parents=True, exist_ok=False)
        self._write(d / "config.json", {"experiment_id": eid, "method": method,
                                        "created_utc": eid.split("-", 1)[0], "config": config,
                                        "config_sha256": sha256_of(config), "meta": meta or {}})
        if record is not None:
            self._write(d / "record.json", record)
        return eid

    def add_circuit(self, eid: str, name: str, circuit: GeneratedCircuit, gen_log=None) -> None:
        self._append(self._dir(eid) / "circuits.jsonl", {"name": name, "circuit": circuit, "gen_log": gen_log})

    def save_grades(self, eid: str, grades: dict) -> None:
        """Merge {circuit name: GradeReport} into grades.json."""
        path = self._dir(eid) / "grades.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        self._write(path, {**current, **to_jsonable(grades)})

    def add_execution(self, eid: str, execution) -> None:
        self._append(self._dir(eid) / "executions.jsonl", execution)

    def save_export(self, eid: str, filename: str, text: str) -> Path:
        if not EXPORT_RE.match(filename) or ".." in filename:
            raise ValueError(f"invalid export file name {filename!r}")
        path = self._dir(eid) / "exports" / filename
        path.write_text(text)
        return path

    def load(self, eid: str) -> dict:
        d = self._dir(eid)
        rec = d / "record.json"
        circuits = self._read_jsonl(d / "circuits.jsonl")
        for c in circuits:
            c["circuit"] = GeneratedCircuit(**c["circuit"])
        grades = d / "grades.json"
        return {
            **json.loads((d / "config.json").read_text()),
            "record": HamiltonianRecord(**json.loads(rec.read_text())) if rec.exists() else None,
            "circuits": circuits,
            "grades": json.loads(grades.read_text()) if grades.exists() else {},
            "executions": self._read_jsonl(d / "executions.jsonl"),
            "exports": sorted(p.name for p in (d / "exports").iterdir() if p.is_file()),
        }

    def read_export(self, eid: str, filename: str) -> str:
        if not EXPORT_RE.match(filename) or ".." in filename:
            raise ValueError(f"invalid export file name {filename!r}")
        return (self._dir(eid) / "exports" / filename).read_text()

    def list(self) -> list[dict]:
        """Summaries, newest first. Directories that are not experiments are ignored."""
        if not self.root.is_dir():
            return []
        out = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if d.is_dir() and ID_RE.match(d.name) and (d / "config.json").exists():
                cfg = json.loads((d / "config.json").read_text())
                out.append({"experiment_id": d.name, "method": cfg.get("method"), "created_utc": cfg.get("created_utc"),
                            "config_sha256": cfg.get("config_sha256"), "meta": cfg.get("meta", {}),
                            "num_circuits": len(self._read_jsonl(d / "circuits.jsonl")),
                            "num_executions": len(self._read_jsonl(d / "executions.jsonl"))})
        return out

    def delete(self, eid: str) -> None:
        shutil.rmtree(self._dir(eid))
