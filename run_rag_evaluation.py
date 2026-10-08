"""Execute the notebook setup and evaluation block without interactive cells."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--check-judge', action='store_true')
    parser.add_argument('--judge-model', default='openai/gpt-oss-20b')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if args.check_judge:
        from dotenv import load_dotenv
        from groq import Groq
        from rag_eval import validate_evaluation, judge_messages, parse_judgment, apply_abstention_checks, JUDGE_RESPONSE_FORMAT
        load_dotenv()
        question = next(q for q in validate_evaluation() if q['id'] == 'q005')
        row = json.loads(Path('rag_evaluation_results.json').read_text(encoding='utf-8'))['results']['q005']
        client = Groq(timeout=60, max_retries=0)
        response = client.chat.completions.create(
            model=args.judge_model, temperature=0, reasoning_effort='low',
            max_completion_tokens=2048, response_format=JUDGE_RESPONSE_FORMAT,
            messages=judge_messages(question, row['answer'], row['context'], row['sources']))
        raw = parse_judgment(response.choices[0].message.content)
        scores, checks = apply_abstention_checks(question, row['answer'], raw)
        print(json.dumps({'raw_judge':raw,'scores':scores,'automatic_checks':checks}, indent=2), flush=True)
        if not scores['abstained'] or scores['unsupported_claims'] or scores['faithfulness'] < 4 or scores['correctness'] > 2:
            raise RuntimeError('Judge still conflates reference material and missing retrieved evidence.')
        return
    # The project's chosen model is already downloaded; prevent remote cache checks.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['ANONYMIZED_TELEMETRY'] = 'False'
    # Bound CPU thread pools for predictable CPU evaluation on this workstation.
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    notebook_path = Path('rag.ipynb')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    count = 0
    with Path('evaluation_run.log').open('a', encoding='utf-8') as log:
        log.write(f'\nRun {datetime.now(timezone.utc).isoformat()}\n')
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            # Compile all code first so notebook ordering/syntax failures are clear.
            compile(code, f'rag.ipynb cell {index}', 'exec')
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            is_evaluation = 'retrieval_results = evaluate_retrieval(' in code
            if args.retrieval_only and is_evaluation:
                code = code.replace('RUN_ANSWER_EVALUATION = True', 'RUN_ANSWER_EVALUATION = False')
            count += 1
            print(f'Executing notebook cell {index}: {code.splitlines()[0][:100]}', flush=True)
            captured = io.StringIO()
            cell['outputs'] = []
            cell['execution_count'] = count
            error = None
            try:
                with redirect_stdout(captured):
                    exec(compile(code, f'rag.ipynb cell {index}', 'exec'), namespace)
            except BaseException as exc:
                error = exc
                cell['outputs'].append({'output_type':'error','ename':type(exc).__name__,
                    'evalue':str(exc),'traceback':traceback.format_exception(exc)})
            output = captured.getvalue()
            if output:
                cell['outputs'].insert(0, {'output_type':'stream','name':'stdout',
                                         'text':output.splitlines(keepends=True)})
                print(output, end='', flush=True)
                log.write(output)
            log.flush()
            notebook_path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
            if error:
                raise error
            if is_evaluation:
                break
    print('Notebook setup and evaluation completed.', flush=True)


if __name__ == '__main__':
    main()
