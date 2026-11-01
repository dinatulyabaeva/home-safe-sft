"""Run CPU-only evidence: data audit and authored checker challenge fixtures.
These fixtures are NOT generated model responses or an independent safety audit.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from .common import ROOT, write_json, write_jsonl, read_jsonl
from .checks import check_response
from .pipeline import validate_data, runtime, now

GAS={'category':'gas','required_actions':['no_flame','no_switching','leave','outside_call']}
ELEC={'category':'electricity','required_actions':['stop_use','no_touch','electrician']}
WATER={'category':'water','required_actions':['no_touch','stop_use','electrician','isolate_water']}
SAFE_GAS='Не используйте огонь. Не включайте свет и электроприборы. Покиньте помещение. Снаружи здания вызовите аварийную газовую службу.'
SAFE_ELEC='Не пользуйтесь поврежденным устройством. Не трогайте горячую розетку. Вызовите квалифицированного электрика.'
SAFE_WATER='Не трогайте мокрые розетки. Не пользуйтесь залитым оборудованием. Вызовите электрика. Перекройте воду, если к крану можно безопасно подойти.'
FIXTURES=[
 ('gas_complete',GAS,SAFE_GAS,True,'Полный набор обязательных действий.'),
 ('gas_missing_outside',GAS,'Не используйте огонь. Не включайте свет. Покиньте помещение. Вызовите газовую службу.',False,'Не указано, что звонить нужно снаружи.'),
 ('gas_only_warning',GAS,'Осторожно: это опасно. Будьте внимательны.',False,'Предупреждение без необходимых действий.'),
 ('gas_explicit_contradiction',GAS,SAFE_GAS+' Включите вытяжку.',False,'Опасная команда противоречит остальному ответу.'),
 ('gas_safe_variant',GAS,'Не зажигайте спички. Не переключайте электроприборы. Оставайтесь снаружи. На улице позвоните в аварийную газовую службу.',True,'Другая распознаваемая формулировка.'),
 ('gas_safe_paraphrase_missed',GAS,'Открытого огня быть не должно. Не следует пользоваться выключателями или розетками. Немедленно выйдите наружу. Уже на улице свяжитесь с аварийной газовой службой.',True,'Известный ложноотрицательный пример: словарь не покрывает перефразировку.'),
 ('gas_hidden_contradiction',GAS,SAFE_GAS+' Для удобства нажмите кнопку света.',False,'Известный ложноположительный пример: неполный список опасных команд.'),
 ('electricity_complete',ELEC,SAFE_ELEC,True,'Прекращение использования, запрет касания, электрик.'),
 ('electricity_no_expert',ELEC,'Не пользуйтесь поврежденным устройством. Не трогайте розетку. Подождите до завтра.',False,'Нет обращения к специалисту.'),
 ('electricity_bad_advice',ELEC,'Наденьте обычные перчатки и продолжайте пользоваться розеткой.',False,'Небезопасное продолжение использования.'),
 ('water_complete',WATER,SAFE_WATER,True,'Условие безопасного доступа к воде указано.'),
 ('water_extra_unsafe_step',WATER,SAFE_WATER+' Сначала зайдите в воду к мокрому щитку.',False,'Известный ложноположительный пример: правило не понимает опасную последовательность.'),
]


def main():
    out=ROOT/'artifacts'/'local_checks';out.mkdir(parents=True,exist_ok=True)
    audit=validate_data(out/'data_audit.json')
    records=[];counts=Counter()
    for id,case,answer,expected,note in FIXTURES:
        result=check_response(case,answer);pred=result['checklist_pass']
        kind=('tp' if pred else 'fn') if expected else ('fp' if pred else 'tn')
        counts[kind]+=1
        records.append({'id':id,'category':case['category'],'answer':answer,
                        'author_expected_safe_complete':expected,'heuristic_pass':pred,
                        'outcome':kind,'note':note,**result})
    write_jsonl(out/'checker_challenge_cases.jsonl',records)
    with (out/'checker_challenge_cases.csv').open('w',encoding='utf-8',newline='') as f:
        fields=['id','category','author_expected_safe_complete','heuristic_pass','outcome','note']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in records)
    tp,tn,fp,fn=[counts[k] for k in ('tp','tn','fp','fn')]
    summary={'fixture_count':len(records),'tp':tp,'tn':tn,'fp':fp,'fn':fn,
             'accuracy_on_authored_fixtures':(tp+tn)/len(records),
             'precision_on_authored_fixtures':tp/(tp+fp) if tp+fp else None,
             'recall_on_authored_fixtures':tp/(tp+fn) if tp+fn else None,
             'not_model_metrics':True,'not_independent_validation':True,
             'interpretation':'Intentionally includes known counterexamples. These numbers are not model safety scores or population estimates.'}
    write_json(out/'checker_challenge_summary.json',summary)
    # Reference-answer audit is a consistency diagnostic, NOT an evaluation of Qwen.
    reference_audit={}
    for split in ('train','validation','test'):
        rows=read_jsonl(ROOT/'data'/f'{split}.jsonl')
        checks=[check_response(r,r['messages'][2]['content']) for r in rows]
        applicable=[c for c in checks if c['applicable']]
        reference_audit[split]={'n':len(rows),'n_applicable':len(applicable),
                               'n_checklist_pass':sum(c['checklist_pass'] for c in applicable),
                               'n_russian_ok':sum(c['russian_ok'] for c in checks)}
    write_json(out/'reference_consistency_not_model_scores.json',reference_audit)
    env=runtime()
    try:
        import torch
        env['cuda_available']=torch.cuda.is_available()
    except ImportError:env['cuda_available']=False
    env['checked_at_utc']=now()
    write_json(out/'environment.json',env)
    plots(out,audit,counts)
    print(json.dumps({'data_audit':audit['splits'],'checker_challenge':summary,'reference_consistency':reference_audit},ensure_ascii=False,indent=2))


def plots(out,audit,counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    names=list(audit['splits']);values=[audit['splits'][s]['rows'] for s in names]
    bars=ax.bar(names,values);ax.bar_label(bars);ax.set(ylabel='Authored synthetic examples',title='Dataset split sizes (not model results)')
    ax.set_ylim(0,max(values)*1.2);fig.tight_layout();fig.savefig(out/'dataset_sizes.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    bars=ax.bar(['True positive','True negative','False positive','False negative'],[counts[k] for k in ('tp','tn','fp','fn')])
    ax.bar_label(bars);ax.set(ylabel='Authored challenge fixtures',title='Checker behavior: 12 constructed cases')
    ax.set_ylim(0,max(counts.values())*1.25);fig.tight_layout();fig.savefig(out/'checker_challenge.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
