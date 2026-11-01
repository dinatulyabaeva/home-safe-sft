"""Paired evaluation with family-cluster bootstrap; never fills missing results."""
from __future__ import annotations
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from .common import ROOT, read_jsonl, write_json, write_jsonl, sha256, unique_index
from .checks import check_response


def quantile(values, q):
    if not values: return None
    values=sorted(values);x=(len(values)-1)*q;lo=int(x);hi=min(lo+1,len(values)-1)
    return values[lo]*(hi-x)+values[hi]*(x-lo) if hi!=lo else values[lo]


def paired_cluster_summary(rows: list[dict], key: str, n_boot=2000, seed=42):
    """Each row: family_id, category, baseline and tuned dicts.
    Resample whole families WITHIN category. Families have equal weight.
    Missing values are excluded only as complete pairs and n is reported.
    """
    groups=defaultdict(list)
    for r in rows:
        a=r['baseline'].get(key);b=r['tuned'].get(key)
        if a is not None and b is not None:
            groups[(r['category'],r['family_id'])].append((float(a),float(b)))
    if not groups:
        return {'metric':key,'n_examples':0,'n_families':0,'baseline':None,'tuned':None,
                'delta':None,'delta_ci95_low':None,'delta_ci95_high':None}
    means={g:(sum(a for a,b in v)/len(v),sum(b for a,b in v)/len(v)) for g,v in groups.items()}
    a=sum(v[0] for v in means.values())/len(means);b=sum(v[1] for v in means.values())/len(means)
    strata=defaultdict(list)
    for (cat,fam),(aa,bb) in means.items():strata[cat].append(bb-aa)
    rng=random.Random(seed);deltas=[]
    for _ in range(n_boot):
        sampled=[rng.choice(vals) for vals in strata.values() for _ in vals]
        deltas.append(sum(sampled)/len(sampled))
    return {'metric':key,'n_examples':sum(map(len,groups.values())),'n_families':len(groups),
            'baseline':a,'tuned':b,'delta':b-a,'delta_ci95_low':quantile(deltas,.025),
            'delta_ci95_high':quantile(deltas,.975)}


def load_verified_predictions(path: Path, data: list[dict], expected_label: str):
    meta_path=path.with_suffix('.meta.json')
    if not path.exists() or not meta_path.exists():raise FileNotFoundError(f'Missing actual predictions or metadata: {path}')
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    if not meta.get('completed'):raise ValueError(f'Incomplete run: {path}')
    if meta['model_label']!=expected_label:raise ValueError('Wrong model label')
    if meta['prediction_sha256']!=sha256(path):raise ValueError('Prediction file changed')
    if meta['dataset_sha256']!=sha256(ROOT/'data'/'test.jsonl'):raise ValueError('Wrong test set')
    rows=read_jsonl(path);idx=unique_index(rows)
    truth=unique_index(data)
    if set(idx)!=set(truth):raise ValueError('Prediction IDs do not cover exactly the frozen test set')
    for id,r in idx.items():
        if r['question']!=truth[id]['messages'][1]['content']:raise ValueError('Question mismatch')
        if r['family_id']!=truth[id]['family_id'] or r['category']!=truth[id]['category']:raise ValueError('Metadata mismatch')
    return idx,meta


def verified_pair(baseline: Path, tuned: Path):
    data=read_jsonl(ROOT/'data'/'test.jsonl')
    a,ma=load_verified_predictions(baseline,data,'baseline')
    b,mb=load_verified_predictions(tuned,data,'tuned')
    for k in ('model_id','model_revision','experiment_lock_sha256','dataset_sha256'):
        if ma[k]!=mb[k]:raise ValueError(f'Non-comparable runs: {k}')
    for k in ('do_sample','num_beams','repetition_penalty','max_new_tokens','eos_token_id','pad_token_id'):
        if ma['generation'].get(k)!=mb['generation'].get(k):raise ValueError(f'Non-comparable decoding: {k}')
    return data,a,b,ma,mb


def evaluate(baseline,tuned):
    data,a,b,ma,mb=verified_pair(Path(baseline),Path(tuned))
    paired=[];details=[]
    for ex in data:
        pair={'id':ex['id'],'family_id':ex['family_id'],'category':ex['category'],'risk':ex['risk']}
        for label,pred,meta in [('baseline',a[ex['id']],ma),('tuned',b[ex['id']],mb)]:
            checks=check_response(ex,pred['answer'])
            metrics={
                'high_risk_action_coverage':checks['coverage'] if ex['risk']=='high' else None,
                'high_risk_checklist_pass':checks['checklist_pass'] if ex['risk']=='high' else None,
                'all_applicable_checklist_pass':checks['checklist_pass'],
                'russian_ok':checks['russian_ok'],
                'mean_word_count':checks['word_count'],
                'repeated_sentence_flag':checks['repeated_sentence_flag'],
                'household_emergency_phrase_flag':checks['emergency_phrase_flag'],
                'token_limit_rate':(pred['generated_tokens']>=meta['generation']['max_new_tokens'] and not pred['ended_with_eos']),
            }
            pair[label]=metrics
            details.append({'id':ex['id'],'model_label':label,**checks})
        paired.append(pair)
    metrics=[paired_cluster_summary(paired,k) for k in paired[0]['baseline']]
    out=ROOT/'artifacts'/'metrics';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'automatic_metrics.json',{'aggregation':'equal weight per scenario family; stratified family bootstrap',
                                            'bootstrap_replicates':2000,'seed':42,'metrics':metrics})
    write_jsonl(out/'per_example_checks.jsonl',details)
    with (out/'automatic_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    by_cat={cat:[paired_cluster_summary([r for r in paired if r['category']==cat],k) for k in paired[0]['baseline']]
            for cat in sorted({r['category'] for r in paired})}
    write_json(out/'by_category.json',by_cat)
    report=['# Фактические результаты автоматической оценки','',
            'Создано только из завершенных запусков. Лексическая проверка не доказывает безопасность.',
            'Агрегация — среднее по семействам; 95% интервал разности получен бутстрэпом семейств внутри категорий.','',
            '| Метрика | До | После | Δ | 95% ДИ Δ | N / семейства |',
            '|---|---:|---:|---:|---|---:|']
    for m in metrics:
        if m['baseline'] is None:continue
        report.append(f"| {m['metric']} | {m['baseline']:.3f} | {m['tuned']:.3f} | {m['delta']:+.3f} | [{m['delta_ci95_low']:.3f}; {m['delta_ci95_high']:.3f}] | {m['n_examples']} / {m['n_families']} |")
    report+=['','## Примеры: фиксированный первый вопрос каждого семейства','',
             'Это не только удачные ответы: выбор определяется ID, а не качеством результата.','']
    seen=set()
    for ex in data:
        if ex['family_id'] in seen:continue
        seen.add(ex['family_id'])
        report += [f"### {ex['id']}",ex['messages'][1]['content'],'',
                   '**До:** '+a[ex['id']]['answer'],'','**После:** '+b[ex['id']]['answer'],'']
    (ROOT/'reports'/'results_generated.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    plot_metrics(metrics)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))


def plot_metrics(metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'artifacts'/'plots';out.mkdir(parents=True,exist_ok=True)
    for m in metrics:
        if m['baseline'] is None:continue
        fig,ax=plt.subplots(figsize=(6.2,4.2))
        ax.bar(['Before SFT','After SFT'],[m['baseline'],m['tuned']])
        ax.set_title(m['metric'].replace('_',' '));ax.set_ylabel('Family-macro mean')
        if m['metric']!='mean_word_count':ax.set_ylim(0,1.05)
        ax.text(.02,.97,f"N={m['n_examples']}; families={m['n_families']}",transform=ax.transAxes,va='top')
        fig.tight_layout();fig.savefig(out/f"{m['metric']}.png",dpi=160);plt.close(fig)
    history_path=ROOT/'artifacts'/'logs'/'train_full_history.json'
    if history_path.exists():
        history=json.loads(history_path.read_text(encoding='utf-8'))
        for key in ('loss','eval_loss'):
            h=[x for x in history if key in x]
            if not h:continue
            fig,ax=plt.subplots(figsize=(6.2,4.2));ax.plot([x['step'] for x in h],[x[key] for x in h],marker='o')
            ax.set(xlabel='Optimizer step',ylabel=key,title='Training log: '+key)
            fig.tight_layout();fig.savefig(out/f'{key}.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',default=str(ROOT/'artifacts'/'baseline_test.jsonl'))
    p.add_argument('--tuned',default=str(ROOT/'artifacts'/'tuned_test.jsonl'))
    a=p.parse_args();evaluate(a.baseline,a.tuned)

if __name__=='__main__':main()
