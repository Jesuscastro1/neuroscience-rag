"""Durable paper records and retrieval history, independent of the search index."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile


def save_json(path, data):
    """Replace a JSON snapshot atomically; leave the previous file intact on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary = file.name
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


class PaperArchive:
    """Commit every observation before further processing; retain changed versions."""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "paper_archive.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    fingerprint TEXT PRIMARY KEY, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY, source TEXT NOT NULL, query TEXT,
                    fingerprint TEXT NOT NULL REFERENCES records(fingerprint),
                    saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, source TEXT NOT NULL, query TEXT,
                    identifier TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                    saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS retrievals (
                    id INTEGER PRIMARY KEY, query TEXT NOT NULL, payload TEXT NOT NULL,
                    saved_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS imports (
                    path TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(path, digest)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _save(db, source, paper, query):
        payload = json.dumps(paper, ensure_ascii=False, sort_keys=True)
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        db.execute("INSERT OR IGNORE INTO records VALUES (?, ?)", (fingerprint, payload))
        db.execute("INSERT INTO observations(source, query, fingerprint) VALUES (?, ?, ?)",
                   (source, query, fingerprint))

    def save(self, source, paper, query=None):
        with self.connect() as db:
            self._save(db, source, paper, query)

    def attempt(self, source, query, identifier, status, error=None):
        with self.connect() as db:
            db.execute("INSERT INTO attempts(source, query, identifier, status, error) VALUES (?, ?, ?, ?, ?)",
                       (source, query, str(identifier), status, str(error) if error else None))

    def record_retrieval(self, query, documents, metadatas):
        payload = json.dumps({"documents": documents, "sources": metadatas}, ensure_ascii=False)
        with self.connect() as db:
            db.execute("INSERT INTO retrievals(query, payload) VALUES (?, ?)", (query, payload))

    def import_json(self, path, source):
        """Import existing corpora once per content version without altering the originals."""
        path = Path(path)
        if not path.exists():
            return
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        with self.connect() as db:
            if db.execute("SELECT 1 FROM imports WHERE path=? AND digest=?",
                          (str(path.resolve()), digest)).fetchone():
                return
            papers = json.loads(content.decode("utf-8"))
            if not isinstance(papers, list) or any(not isinstance(p, dict) for p in papers):
                raise ValueError(f"Expected a list of paper objects in {path}")
            for paper in papers:
                self._save(db, source, paper, f"Imported from {path}")
            db.execute("INSERT INTO imports VALUES (?, ?)", (str(path.resolve()), digest))

    def papers(self, source=None):
        with self.connect() as db:
            if source is None:
                rows = db.execute("SELECT payload FROM records ORDER BY rowid")
            else:
                rows = db.execute("""SELECT r.payload FROM records r WHERE EXISTS
                    (SELECT 1 FROM observations o WHERE o.fingerprint=r.fingerprint AND o.source=?)
                    ORDER BY r.rowid""", (source,))
            return [json.loads(row[0]) for row in rows]

    def export(self):
        """Rebuild readable snapshots from the authoritative archive, including older sources."""
        with self.connect() as db:
            sources = [row[0] for row in db.execute("SELECT DISTINCT source FROM observations")]
        for source in sources:
            if source in {"semantic_scholar", "arxiv", "pubmed"}:
                save_json(self.data_dir / "raw" / f"{source}.json", self.papers(source))
        papers = self.papers()
        save_json(self.data_dir / "merged_papers.json", papers)
        return papers
