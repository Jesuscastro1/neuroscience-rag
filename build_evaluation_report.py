"""Combine evaluation artifacts with plain-language metric explanations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from paper_archive import save_json

METRIC_MEANINGS = {
    'mean_corpus_coverage': 'Fraction of designated supporting papers present in the index. Low coverage points to collection, abstract availability, or indexing problems.',
    'recall_at_k': 'Average fraction of designated supporting papers retrieved in the first k results, over answerable questions. Multi-paper questions require more than one hit.',
    'hit_at_k': 'Fraction of answerable questions retrieving at least one designated supporting paper within k results. A hit alone does not guarantee enough evidence for a complete answer.',
    'ndcg_at_k': 'Rank-sensitive retrieval quality from 0 to 1; designated supporting papers near the top score better. Relevance labels are not exhaustive.',
    'mrr_at_max_k': 'Average inverse rank of the first designated supporting paper, searched through the largest evaluated k. Higher values mean useful evidence appears earlier.',
    'faithfulness': 'Automated score from 1 to 5 for support of factual claims in the actual retrieved abstracts. A justified abstention can be faithful even when the full corpus has an answer.',
    'correctness': 'Automated score from 1 to 5 for accuracy and completeness against the full-corpus reference answer. Pure abstention on an answerable question receives 1.',
    'relevance': 'Automated score from 1 to 5 for addressing the question directly.',
    'citation_quality': 'Automated score from 1 to 5 for citing the correct retrieved paper titles for factual claims. Pure justified abstentions need no citations.',
    'answerability_handling': 'Automated score from 1 to 5 for answering only what the retrieved evidence supports and acknowledging missing information.',
    'overall': 'Automated summary score from 1 to 5; it is not a percentage accuracy measure. Pure abstention on an answerable item is capped at 2.',
    'unanswerable_abstention_rate': 'Fraction of scored, globally unanswerable questions on which the answer explicitly abstains. Higher is better for these cases.',
    'answerable_abstention_rate': 'Fraction of scored, globally answerable questions on which the answer abstains. This can expose missing retrieved evidence or over-abstention.',
    'unsupported_claim_rate': 'Fraction of scored answers flagged by the judge as making claims not supported by the retrieved abstracts. Lower is better.',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build_report(output_path='evaluation_report.json'):
    dataset = read_json('evaluation.json')
    retrieval = read_json('retrieval_evaluation_results.json')
    answers = read_json('rag_evaluation_results.json')
    corpus = read_json('data/merged_papers.json')
    questions = dataset['questions'] if isinstance(dataset, dict) else dataset
    config = {'questions': questions, 'index_fingerprint': retrieval['index_fingerprint'],
              'answer_model': answers.get('answer_model'), 'top_k': answers.get('top_k')}
    expected = hashlib.sha256(json.dumps(config,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    matches = answers.get('answer_fingerprint') == expected
    result_rows = answers.get('results', {})
    all_rows_present = all(
        str(q['id']) in result_rows
        and 'answer' in result_rows[str(q['id'])]
        and 'scores' in result_rows[str(q['id'])]
        and not result_rows[str(q['id'])].get('error')
        for q in questions
    )
    complete = bool(matches and all_rows_present and answers.get('summary', {}).get('complete'))
    summary = answers.get('summary', {})
    rs = retrieval['summary']
    top_k = answers.get('top_k', 5)
    missing = [row for row in retrieval['per_question'] if row['recall_at_k'].get(str(top_k), 0) < 1]
    findings = [
        f"The index contains {retrieval['indexed_abstracts']} distinct abstract records. Designated-source corpus coverage is {rs['mean_corpus_coverage']:.1%}.",
        f"Recall at {top_k} is {rs['recall_at_k'][str(top_k)]:.2%}. {len(missing)} answerable questions lack at least one designated source within this retrieval budget.",
        f"Recall at 10 is {rs['recall_at_k']['10']:.2%} on this development set.",
    ]
    if not matches:
        findings.append('Answer results do not match the current dataset/index; rerun evaluation before interpreting answer scores.')
    elif not complete:
        findings.append(f"All available results are checkpointed. {summary.get('num_answered',0)}/{len(questions)} answers and {summary.get('num_scored',0)}/{len(questions)} judgments are saved. Answer-quality averages are provisional.")
    else:
        findings.append(f"All {len(questions)} answers were generated and judged. Mean correctness is {summary['averages']['correctness']:.2f}/5; mean faithfulness is {summary['averages']['faithfulness']:.2f}/5.")
        findings.append(f"The unsupported-question abstention rate is {summary['unanswerable_abstention_rate']:.1%}; the judge's unsupported-claim rate is {summary['unsupported_claim_rate']:.1%}.")
    low_scores = [dict(id=r['id'],question=r['question'],scores=r['scores'],reason=r['scores']['reason'])
                  for r in answers.get('results', {}).values()
                  if 'scores' in r and (r['scores']['correctness'] <= 3 or r['scores']['faithfulness'] <= 3 or r['scores']['unsupported_claims'])]
    report = {
        'schema_version': 1, 'generated_at': datetime.now(timezone.utc).isoformat(),
        'status': 'complete' if complete else ('partial' if matches else 'stale'),
        'answer_results_are_final': complete,
        'run_matches_current_dataset_and_index': matches,
        'dataset': {'path':'evaluation.json','num_questions':len(questions),
                    'num_answerable':sum(q['answerable'] for q in questions),
                    'num_unanswerable':sum(not q['answerable'] for q in questions),
                    'questions':questions},
        'corpus': {'path':'data/merged_papers.json','archived_record_versions':len(corpus),
                   'record_versions_with_abstracts':sum(bool((p.get('abstract') or '').strip()) for p in corpus),
                   'indexed_abstracts':retrieval['indexed_abstracts']},
        'models': {'embedding_model':retrieval['embedding_model'],
                   'answer_model':answers.get('answer_model'),'judge_model':answers.get('judge_model'),
                   'answer_retrieval_top_k':top_k},
        'execution_order':['validate questions and evidence','build/load matching paper index',
                           'evaluate retrieval','generate and save answers',
                           'judge saved answers','apply deterministic abstention checks','summarize results'],
        'retrieval':retrieval, 'answer_evaluation':answers,
        'interpretation': {'metric_meanings':METRIC_MEANINGS,'findings':findings,
            'retrieval_cases_to_review':[dict(id=r['id'],question=r['question'],
                recall_at_k=r['recall_at_k'],expected_sources=r['expected_sources'],
                ranked_sources=r['ranked_sources']) for r in missing],
            'answer_cases_to_review':low_scores,
            'next_steps':[
                'Try top_k=10 and rerun answer evaluation: designated-source recall is higher at 10 on this dataset, but answer quality still needs to be measured.',
                'Review low-correctness, low-faithfulness, or unsupported-claim cases alongside their saved contexts and raw judge scores.',
                'Use an independently held-out, human-reviewed question set before treating these development results as general model performance.'
            ],
            'limitations':[
                'These questions were authored from this corpus and are a development set, not an independent held-out benchmark.',
                'Gold source labels designate supporting papers but do not exhaustively label every relevant paper.',
                'Answer grades are automated estimates, not human adjudication.',
                'The 20B judge evaluates saved answers from the 120B model; changing the judge can change subjective scores.',
                'Raw judge scores and deterministic adjustments are preserved for audit.'
            ]}
    }
    save_json(output_path,report)
    print(f"Saved {output_path}: {report['status']}")
    return report


if __name__=='__main__':
    build_report()
