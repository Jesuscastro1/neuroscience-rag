import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rag_eval import validate_evaluation, retrieval_metrics, parse_judgment, run_answer_evaluation, judge_messages, apply_abstention_checks, SCORE_NAMES


class EvaluationTests(unittest.TestCase):
    def test_all_questions_have_verifiable_saved_evidence(self):
        questions = validate_evaluation()
        self.assertEqual(len(questions), 38)
        self.assertEqual(sum(q['answerable'] for q in questions), 32)
        self.assertEqual(len({q['id'] for q in questions}), 38)
        self.assertEqual(sum(q['category']=='multi_paper' for q in questions), 3)

    def test_retrieval_metrics_known_ranks_and_negative_exclusion(self):
        questions = [dict(id='one', question='Q', category='multi_paper', answerable=True,
                          relevant_papers=['A','B']),
                     dict(id='negative', question='No answer', answerable=False)]
        results = {'metadatas':[[{'source':'noise'},{'source':'A'},{'source':'B'}]]}
        scores = retrieval_metrics(questions, results, ['A','B','noise'], str.lower, ks=(1,2,3))
        summary = scores['summary']
        self.assertEqual(summary['recall_at_k'], {'1':0.0,'2':0.5,'3':1.0})
        self.assertEqual(summary['mrr_at_max_k'], 0.5)
        self.assertEqual(summary['unanswerable_questions_excluded'], 1)
        self.assertEqual(summary['mean_corpus_coverage'], 1.0)
        self.assertLess(summary['ndcg_at_k']['3'], 1.0)

    def test_partial_corpus_coverage_not_mistaken_for_retrieval_failure(self):
        questions = [dict(id='one', question='Q', category='multi_paper', answerable=True,
                          relevant_papers=['A','B'])]
        result = retrieval_metrics(questions, {'metadatas':[[{'source':'A'}]]}, ['A'], str.lower, ks=(5,))
        self.assertEqual(result['summary']['mean_corpus_coverage'], 0.5)
        self.assertEqual(result['summary']['recall_at_k']['5'], 0.5)
        self.assertEqual(result['summary']['conditional_recall_at_k']['5'], 1.0)

    def test_judge_scores_must_be_valid_not_arbitrary_json(self):
        scores = {name:5 for name in SCORE_NAMES}
        scores.update(abstained=False, unsupported_claims=False, reason='Supported')
        self.assertEqual(parse_judgment(json.dumps(scores)), scores)
        scores['faithfulness'] = 8
        with self.assertRaises(ValueError):
            parse_judgment(json.dumps(scores))

    def test_answers_saved_then_judged_and_resume_does_not_repeat_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'results.json'
            questions = [dict(id=str(i),question=f'Q{i}',category='factual',answerable=True,
                              relevant_papers=['A'],expected_answer='Answer',evidence=[]) for i in range(2)]
            retrieve = Mock(return_value=(['abstract'],[{'source':'A'}]))
            calls = []
            def generate(context, question, **kwargs):
                calls.append(('answer', question))
                return 'Answer [A]'
            scores = {name:5 for name in SCORE_NAMES}
            scores.update(abstained=False, unsupported_claims=False,reason='Supported')
            def judge(messages):
                calls.append(('judge', ''))
                return json.dumps(scores)
            state = run_answer_evaluation(questions,retrieve,generate,judge,'fingerprint',output_path=output)
            self.assertEqual([c[0] for c in calls], ['answer','answer','judge','judge'])
            self.assertTrue(state['summary']['complete'])
            run_answer_evaluation(questions,retrieve,generate,judge,'fingerprint',output_path=output)
            self.assertEqual(len(calls), 4)

    def test_api_failure_checkpoints_answer_and_retrieved_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'results.json'
            questions = [dict(id=str(i),question=f'Q{i}',category='factual',answerable=True,
                              relevant_papers=['A'],expected_answer='Answer',evidence=[]) for i in range(2)]
            retrieve = Mock(return_value=(['abstract'],[{'source':'A'}]))
            generate = Mock(side_effect=['Answer',RuntimeError('Rate limited')])
            judge = Mock()
            with self.assertRaises(RuntimeError):
                run_answer_evaluation(questions,retrieve,generate,judge,'fingerprint',output_path=output)
            state = json.loads(output.read_text())
            self.assertEqual(state['results']['0']['answer'], 'Answer')
            self.assertEqual(state['results']['1']['context'], ['abstract'])
            self.assertFalse(state['summary']['complete'])
            judge.assert_not_called()

    def test_judge_never_receives_reference_quotes_as_retrieved_evidence(self):
        item = dict(question='Why?',expected_answer='Reference answer',answerable=True,
                    relevant_papers=['Missing source'],
                    evidence=[{'paper_title':'Missing source','quote':'REFERENCE_QUOTE_SENTINEL'}])
        messages = judge_messages(item, 'This question is not addressed in the provided papers.',
                                  ['Unrelated abstract'], [{'source':'Other source'}])
        payload = json.loads(messages[1]['content'])
        self.assertNotIn('REFERENCE_QUOTE_SENTINEL', messages[1]['content'])
        self.assertTrue(payload['GENERATED_ANSWER_IS_EXPLICIT_ABSTENTION'])
        self.assertEqual(payload['ACTUAL_RETRIEVED_EVIDENCE']['abstracts'], ['Unrelated abstract'])
        self.assertEqual(payload['REFERENCE_FOR_CORRECTNESS_ONLY']['designated_source_titles_missing'], ['Missing source'])

    def test_changed_judge_reuses_answers_and_rejudges(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'results.json'
            questions = [dict(id='one',question='Q',category='factual',answerable=True,
                              relevant_papers=['A'],expected_answer='Answer',evidence=[])]
            retrieve = Mock(return_value=(['abstract'],[{'source':'A'}]))
            generate = Mock(return_value='Answer')
            scores = {name:5 for name in SCORE_NAMES}
            scores.update(abstained=False, unsupported_claims=False,reason='Supported')
            judge = Mock(return_value=json.dumps(scores))
            run_answer_evaluation(questions,retrieve,generate,judge,'fingerprint',output_path=output)
            run_answer_evaluation(questions,retrieve,generate,judge,'fingerprint',output_path=output,judge_model='different-judge')
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(judge.call_count, 2)

    def test_pure_abstention_uses_known_answerability_label_for_correctness(self):
        raw = {name:5 for name in SCORE_NAMES}
        raw.update(abstained=False, unsupported_claims=False,reason='LLM grade')
        text = 'This question is not addressed in the provided papers.'
        scores, checks = apply_abstention_checks({'answerable':True},text,raw)
        self.assertTrue(scores['abstained'])
        self.assertEqual(scores['correctness'],1)
        self.assertEqual(scores['overall'],2)
        self.assertEqual(scores['faithfulness'],5)
        self.assertTrue(checks['judge_score_adjusted'])
        self.assertEqual(raw['correctness'],5)  # Preserve the unmodified judge output for review.
        scores, _ = apply_abstention_checks({'answerable':False},text,raw)
        self.assertEqual(scores['correctness'],5)

    def test_notebook_sequence_compiles_and_definitions_precede_execution(self):
        n=json.loads(Path('rag.ipynb').read_text(encoding='utf-8'))
        definition = execution = None
        for i,c in enumerate(n['cells']):
            if c['cell_type']=='code':
                code=''.join(c['source'])
                compile(code,'rag.ipynb','exec')
                if 'def evaluate_answers(' in code: definition=i
                if 'retrieval_results = evaluate_retrieval(' in code: execution=i
        self.assertLess(definition, execution)
        code=''.join(n['cells'][execution]['source'])
        self.assertLess(code.index('evaluate_retrieval('),code.index('evaluate_answers('))


if __name__=='__main__':
    unittest.main()
