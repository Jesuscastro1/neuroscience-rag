"""Offline persistence regressions; no paper APIs, model downloads, or LLM calls."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper_archive import PaperArchive, save_json

NOTEBOOK = json.loads(Path('rag.ipynb').read_text(encoding='utf-8'))

def notebook_function(name):
    for cell in NOTEBOOK['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == name:
                    return compile(ast.Module(body=[node], type_ignores=[]), 'rag.ipynb', 'exec')
    raise AssertionError(f'Missing function: {name}')


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = PaperArchive(self.root / 'data')

    def namespace(self):
        import hashlib
        import unicodedata
        import sqlite3
        namespace = dict(PAPER_ARCHIVE=self.archive, save_json=save_json,
                         MERGED_PATH=self.root/'data'/'merged_papers.json',
                         time=SimpleNamespace(sleep=lambda _: None), os=__import__('os'),
                         hashlib=hashlib, unicodedata=unicodedata, sqlite3=sqlite3)
        exec(notebook_function('checkpoint_fetch'), {'wraps': __import__('functools').wraps,
             'PAPER_ARCHIVE': self.archive}, namespace)
        return namespace

    def test_restart_versions_missing_abstract_and_repeated_observations(self):
        original = {'title': 'Brain', 'abstract': None, 'pmid': '1'}
        updated = dict(original, abstract='Complete abstract')
        for paper in [original, updated, updated, {'title': None, 'abstract': 'Untitled'}]:
            self.archive.save('pubmed', paper, 'brain')
        restarted = PaperArchive(self.archive.data_dir)
        papers = restarted.export()
        self.assertEqual(len(papers), 3)
        self.assertIn(original, papers)
        self.assertIn(updated, papers)
        with restarted.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 4)

    def test_import_all_sources_and_old_merged_without_erasing_them(self):
        old = [{'title':'Old', 'abstract':'Old abstract'}]
        path = self.root/'legacy.json'
        save_json(path, old)
        self.archive.import_json(path, 'legacy_merged')
        self.archive.import_json(path, 'legacy_merged')
        self.archive.save('pubmed', {'title':'PubMed', 'abstract': None})
        self.archive.save('arxiv', {'title':'New', 'abstract':'New abstract'})
        merged = self.archive.export()
        self.assertEqual(len(merged), 3)
        self.assertEqual(json.loads(path.read_text()), old)
        self.assertTrue((self.archive.data_dir/'raw'/'pubmed.json').exists())
        with self.archive.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM imports').fetchone()[0], 1)

    def test_atomic_json_failure_keeps_previous_snapshot(self):
        path = self.root/'snapshot.json'
        save_json(path, ['old'])
        with patch('paper_archive.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError):
                save_json(path, ['new'])
        self.assertEqual(json.loads(path.read_text()), ['old'])
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_arxiv_interruption_retains_already_fetched_record(self):
        ns = self.namespace()
        first = SimpleNamespace(title='Paper', summary='Abstract', entry_id='arxiv:1',
                                published='2026', authors=[])
        def results(_):
            yield first
            raise KeyboardInterrupt()
        ns['arxiv'] = SimpleNamespace(
            Client=lambda **_: SimpleNamespace(results=results), Search=lambda **_: None,
            SortCriterion=SimpleNamespace(SubmittedDate=None),
            SortOrder=SimpleNamespace(Descending=None))
        exec(notebook_function('fetch_arxiv_papers'), ns)
        with self.assertRaises(KeyboardInterrupt):
            ns['fetch_arxiv_papers']('brain', 2)
        self.assertEqual(self.archive.papers()[0]['abstract'], 'Abstract')
        self.assertEqual(len(json.loads((self.archive.data_dir/'merged_papers.json').read_text())), 1)

    def test_semantic_scholar_page_saved_before_next_request_fails(self):
        ns = self.namespace()
        def get(*args, **kwargs):
            if get.called:
                raise KeyboardInterrupt()
            get.called = True
            return SimpleNamespace(status_code=200, json=lambda: {'data':[
                {'paperId':'1', 'title':'One', 'abstract':'Abstract'},
                {'paperId':'2', 'title':'Two', 'abstract':None}]})
        get.called = False
        ns['requests'] = SimpleNamespace(get=get, exceptions=SimpleNamespace(RequestException=RuntimeError))
        exec(notebook_function('fetch_papers'), ns)
        with self.assertRaises(KeyboardInterrupt):
            ns['fetch_papers']('brain', 4, batch_size=2)
        self.assertEqual(len(self.archive.papers('semantic_scholar')), 2)

    def test_pubmed_failed_ids_logged_successful_abstract_saved(self):
        ns = self.namespace()
        def article(pmid):
            if pmid == '2':
                raise RuntimeError('API unavailable')
            return SimpleNamespace(title='One', abstract='Saved', year=2026, authors=[])
        ns['fetch'] = SimpleNamespace(pmids_for_query=lambda *a, **k:['1','2'], article_by_pmid=article)
        exec(notebook_function('fetch_pubmed_papers'), ns)
        self.assertEqual(len(ns['fetch_pubmed_papers']('brain', 2, max_retries=1)), 1)
        with self.archive.connect() as db:
            self.assertEqual(db.execute("SELECT status FROM attempts WHERE identifier='2' ORDER BY id").fetchall(),
                             [('discovered',), ('failed',)])

    def test_pubmed_storage_failure_propagates(self):
        ns = self.namespace()
        ns['fetch'] = SimpleNamespace(pmids_for_query=lambda *a, **k:['1'],
            article_by_pmid=lambda _: SimpleNamespace(title='One', abstract='Saved', year=2026, authors=[]))
        exec(notebook_function('fetch_pubmed_papers'), ns)
        with patch.object(self.archive, 'save', side_effect=OSError('Disk full')):
            with self.assertRaises(OSError):
                ns['fetch_pubmed_papers']('brain', 1)

    def test_title_dedup_prefers_complete_abstract_without_deleting_archive(self):
        ns = self.namespace()
        exec(notebook_function('normalize_title'), ns)
        exec(notebook_function('deduplicate_papers'), ns)
        papers = [{'title':'Brain', 'abstract':None}, {'title':'Brain', 'abstract':'Complete'}]
        for p in papers:
            self.archive.save('arxiv', p)
        self.assertEqual(ns['deduplicate_papers'](papers), [papers[1]])
        self.assertEqual(len(self.archive.papers()), 2)

    def test_retrieved_evidence_saved_before_generation(self):
        ns = self.namespace()
        ns['model'] = SimpleNamespace(encode_query=lambda *a, **k:SimpleNamespace(tolist=lambda:[1.0]))
        ns['collection'] = SimpleNamespace(count=lambda:1,
            query=lambda **k:{'documents':[['Title: One\n\nAbstract: Saved']],
                              'metadatas':[[{'source':'One', 'url':'url'}]]})
        exec(notebook_function('relevant_chunks'), ns)
        ns['relevant_chunks']('What is neuroscience?')
        with self.archive.connect() as db:
            query, payload = db.execute('SELECT query, payload FROM retrievals').fetchone()
        self.assertEqual(query, 'What is neuroscience?')
        self.assertIn('Abstract: Saved', json.loads(payload)['documents'][0])

    def test_all_notebook_code_compiles_and_saving_cell_is_executable(self):
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), 'rag.ipynb', 'exec')
        self.assertTrue(any(cell['cell_type']=='code' and
                            'def fetch_and_save_papers(' in ''.join(cell['source'])
                            for cell in NOTEBOOK['cells']))


if __name__ == '__main__':
    unittest.main()
