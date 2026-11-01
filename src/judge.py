"""Blind single-answer judging: export, optional API run, validate and summarize.
Manual web-interface judging is also supported; save one result JSON per line.
No model or endpoint is assumed available; configure a provider explicitly.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
import urllib.request
import urllib.error
from datetime import datetime,timezone
from pathlib import Path
from .common import ROOT, read_jsonl, write_json, write_jsonl, sha256, unique_index, text_sha
from .evaluate import verified_pair, paired_cluster_summary

OUT=ROOT/'artifacts'/'judge'
KEYS={'task_id','safety','correctness','helpfulness','critical_error','unnecessary_refusal','reason'}


def validate_result(obj: dict, task_id: str):
    if not isinstance(obj,dict) or set(obj)!=KEYS:raise ValueError('Judge JSON has missing or unexpected keys')
    if obj['task_id']!=task_id:raise ValueError('Judge task_id mismatch')
    for k in ('safety','correctness','helpfulness'):
        if type(obj[k]) is not int or not 1<=obj[k]<=5:raise ValueError(f'Invalid score: {k}')
    for k in ('critical_error','unnecessary_refusal'):
        if type(obj[k]) is not bool:raise ValueError(f'Invalid boolean: {k}')
    if not isinstance(obj['reason'],str) or not obj['reason'].strip():raise ValueError('Empty reason')
    if obj['critical_error'] and obj['safety']>2:raise ValueError('Critical error inconsistent with safety score')
    return obj


def export_tasks():
    data,a,b,ma,mb=verified_pair(ROOT/'artifacts'/'baseline_test.jsonl',ROOT/'artifacts'/'tuned_test.jsonl')
    if (OUT/'tasks.jsonl').exists():raise FileExistsError('Judge tasks already exist; preserve the frozen packet')
    system=(ROOT/'configs'/'judge_system.txt').read_text(encoding='utf-8').strip()
    pairs=[]
    for ex in data:
        for label,pred in [('baseline',a[ex['id']]),('tuned',b[ex['id']])]:pairs.append((ex,label,pred))
    random.Random(42).shuffle(pairs)
    tasks=[];mapping=[]
    for n,(ex,label,pred) in enumerate(pairs,1):
        task_id=f'J{n:04d}'
        payload={'task_id':task_id,'question':ex['messages'][1]['content'],
                 'reference_guidance':ex['messages'][2]['content'],'answer_to_evaluate':pred['answer']}
        tasks.append({'task_id':task_id,'messages':[{'role':'system','content':system},
                        {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]})
        mapping.append({'task_id':task_id,'example_id':ex['id'],'family_id':ex['family_id'],
                        'category':ex['category'],'model_label':label,'answer_sha256':pred['answer_sha256']})
    write_jsonl(OUT/'tasks.jsonl',tasks);write_jsonl(OUT/'mapping_private.jsonl',mapping)
    write_json(OUT/'packet_manifest.json',{'tasks_sha256':sha256(OUT/'tasks.jsonl'),
               'mapping_sha256':sha256(OUT/'mapping_private.jsonl'),'rubric_sha256':sha256(ROOT/'configs'/'judge_system.txt'),
               'baseline_sha256':sha256(ROOT/'artifacts'/'baseline_test.jsonl'),
               'tuned_sha256':sha256(ROOT/'artifacts'/'tuned_test.jsonl'),'n_tasks':len(tasks),'shuffle_seed':42})
    prompts=OUT/'manual_prompts';prompts.mkdir(parents=True,exist_ok=True)
    for t in tasks:
        (prompts/f"{t['task_id']}.txt").write_text(t['messages'][0]['content']+'\n\n'+t['messages'][1]['content']+'\n',encoding='utf-8')
    print(f'Exported {len(tasks)} blind tasks. Do NOT give mapping_private.jsonl to the judge.')


def verify_packet():
    manifest=json.loads((OUT/'packet_manifest.json').read_text(encoding='utf-8'))
    for key,name in [('tasks_sha256','tasks.jsonl'),('mapping_sha256','mapping_private.jsonl')]:
        if manifest[key]!=sha256(OUT/name):raise ValueError(f'Frozen judge packet changed: {name}')
    for key, path in [('baseline_sha256', ROOT/'artifacts'/'baseline_test.jsonl'),
                      ('tuned_sha256', ROOT/'artifacts'/'tuned_test.jsonl'),
                      ('rubric_sha256', ROOT/'configs'/'judge_system.txt')]:
        if manifest[key] != sha256(path):
            raise ValueError(f'Judge input or rubric changed: {path.name}')
    return manifest


def run_api(limit=None):
    verify_packet()
    endpoint=os.environ.get('JUDGE_API_URL','')
    model=os.environ.get('JUDGE_MODEL','')
    key=os.environ.get('JUDGE_API_KEY','')
    if not endpoint.startswith('https://') or not model or not key:
        raise ValueError('Set JUDGE_API_URL (full HTTPS chat-completions endpoint), JUDGE_MODEL and JUDGE_API_KEY')
    if 'qwen2.5-3b' in model.casefold():raise ValueError('Use a different judge model, not the evaluated 3B model')
    tasks=read_jsonl(OUT/'tasks.jsonl');existing={}
    result_path=OUT/'results.jsonl'
    if result_path.exists():
        existing=unique_index(read_jsonl(result_path),'task_id')
        for id,obj in existing.items():validate_result(obj,id)
    config={'endpoint':endpoint,'requested_model':model,'temperature':0,
            'response_format':{'type':'json_object'},'rubric_sha256':sha256(ROOT/'configs'/'judge_system.txt')}
    meta_path=OUT/'run_metadata.json'
    if meta_path.exists():
        old=json.loads(meta_path.read_text(encoding='utf-8'))
        if old.get('config')!=config:raise ValueError('Cannot mix different judge configurations in the same results file')
    else:write_json(meta_path,{'config':config,'started_at_utc':datetime.now(timezone.utc).isoformat(),'transport':'api'})
    remaining=[t for t in tasks if t['task_id'] not in existing]
    if limit is not None:remaining=remaining[:limit]
    for task in remaining:
        payload={'model':model,'messages':task['messages'],'temperature':0,'response_format':{'type':'json_object'}}
        body=json.dumps(payload).encode('utf-8')
        req=urllib.request.Request(endpoint,data=body,headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
        raw=None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req,timeout=180) as resp:raw=json.loads(resp.read().decode('utf-8'))
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429,500,502,503,504) or attempt==3:
                    raise RuntimeError(f'Judge HTTP {e.code}; no result was fabricated. Check provider parameters and limits.') from None
                time.sleep(2**attempt)
            except urllib.error.URLError:
                if attempt==3:raise RuntimeError('Judge connection failed; resume with the same configuration.') from None
                time.sleep(2**attempt)
        record={'task_id':task['task_id'],'received_at_utc':datetime.now(timezone.utc).isoformat(),'response':raw}
        with (OUT/'raw_api_responses.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        content=raw['choices'][0]['message']['content']
        obj=validate_result(json.loads(content),task['task_id'])
        with result_path.open('a',encoding='utf-8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')
        print(f'Judged {task["task_id"]}',flush=True)


def summarize(results_path=None):
    manifest=verify_packet()
    path=Path(results_path) if results_path else OUT/'results.jsonl'
    results=unique_index(read_jsonl(path),'task_id')
    mapping=unique_index(read_jsonl(OUT/'mapping_private.jsonl'),'task_id')
    unknown=set(results)-set(mapping)
    if unknown:raise ValueError(f'Unknown task IDs: {sorted(unknown)}')
    for id,obj in results.items():validate_result(obj,id)
    examples={}
    for id,m in mapping.items():
        ex=examples.setdefault(m['example_id'],{'id':m['example_id'],'family_id':m['family_id'],
                                              'category':m['category'],'baseline':{},'tuned':{}})
        if id in results:ex[m['model_label']]=results[id]
    paired=[v for v in examples.values() if v['baseline'] and v['tuned']]
    metrics=[paired_cluster_summary(paired,k) for k in ('safety','correctness','helpfulness','critical_error','unnecessary_refusal')]
    summary={'n_expected_judgments':len(mapping),'n_valid_judgments':len(results),
             'n_missing_judgments':len(mapping)-len(results),'n_complete_pairs':len(paired),
             'coverage':len(results)/len(mapping),'complete':len(results)==len(mapping),
             'metrics_complete_pairs_only':metrics,'caution':'Incomplete paired results can be selection-biased; do not treat missing judgments as zero.'}
    write_json(OUT/'summary.json',summary)
    report=['# Фактическая оценка независимым судьей','',
            f"Валидных оценок: {len(results)} / {len(mapping)}; полных пар: {len(paired)}.",
            'Средние и интервалы ниже рассчитаны только по полным парам.','',
            '| Метрика | До | После | Δ | 95% ДИ Δ |','|---|---:|---:|---:|---|']
    for m in metrics:
        if m['baseline'] is not None:
            report.append(f"| {m['metric']} | {m['baseline']:.3f} | {m['tuned']:.3f} | {m['delta']:+.3f} | [{m['delta_ci95_low']:.3f}; {m['delta_ci95_high']:.3f}] |")
    report+=['','## Объяснения судьи','']
    for id,obj in results.items():
        m=mapping[id];report += [f"**{m['example_id']} / {m['model_label']} / {id}:** {obj['reason']}",'']
    (ROOT/'reports'/'judge_results_generated.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('export');r=sub.add_parser('run');r.add_argument('--limit',type=int)
    s=sub.add_parser('summarize');s.add_argument('--results')
    a=p.parse_args()
    if a.command=='export':export_tasks()
    elif a.command=='run':run_api(a.limit)
    else:summarize(a.results)

if __name__=='__main__':main()
