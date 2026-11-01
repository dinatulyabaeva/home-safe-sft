"""CLI: validate -> prepare -> generate baseline -> train -> generate tuned.
GPU tasks deliberately fail on CPU instead of silently changing the experiment.
"""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from .common import ROOT, read_jsonl, write_json, sha256, text_sha, unique_index
from .encoding import encode_example, CompletionCollator

CONFIG = ROOT/'configs'/'experiment.json'
LOCK = ROOT/'artifacts'/'experiment_lock.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def data_hashes():
    paths = [ROOT/'data'/f'{s}.jsonl' for s in ('train','validation','test','demo_prompts')]
    paths += [CONFIG,ROOT/'configs'/'system_prompt.txt',ROOT/'requirements.txt',ROOT/'requirements-torch.txt']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def validate_data(out: Path | None = None):
    system=(ROOT/'configs'/'system_prompt.txt').read_text(encoding='utf-8').strip()
    families={}; prompts={}; ids=set(); answers={}; summary={}
    from .checks import ACTION_PATTERNS
    for split in ('train','validation','test'):
        rows=read_jsonl(ROOT/'data'/f'{split}.jsonl')
        if not rows: raise ValueError(f'Empty split: {split}')
        for row in rows:
            if row['id'] in ids: raise ValueError(f'Duplicate id: {row["id"]}')
            ids.add(row['id'])
            if row['split'] != split: raise ValueError('Incorrect split metadata')
            roles=[m['role'] for m in row['messages']]
            if roles!=['system','user','assistant']: raise ValueError(f'Invalid roles: {row["id"]}')
            if row['messages'][0]['content']!=system: raise ValueError('System prompt drift')
            if any(not m['content'].strip() for m in row['messages']): raise ValueError('Empty message')
            if set(row['required_actions'])-ACTION_PATTERNS.keys(): raise ValueError('Unknown checklist action')
            if row['risk'] not in ('low','high'): raise ValueError('Invalid risk')
            fam=row['family_id']; q=' '.join(row['messages'][1]['content'].casefold().split())
            ans=' '.join(row['messages'][2]['content'].casefold().split())
            for index,key,label in ((families,fam,'family'),(prompts,q,'prompt'),(answers,ans,'answer')):
                if key in index and index[key]!=split:
                    raise ValueError(f'Cross-split exact {label} duplicate: {row["id"]}')
                index[key]=split
        summary[split]={'rows':len(rows),'families':len({r['family_id'] for r in rows}),
                        'categories':dict(Counter(r['category'] for r in rows)),
                        'risk':dict(Counter(r['risk'] for r in rows))}
    result={'status':'passed','checked_at_utc':now(),'splits':summary,
            'exact_cross_split_family_prompt_answer_leaks':0,
            'semantic_leakage_not_ruled_out':True,'hashes':data_hashes()}
    if out: write_json(out,result)
    return result


def gpu_required():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not available. Select a GPU runtime in Colab. This command does not run a CPU substitute.')
    props=torch.cuda.get_device_properties(0)
    if props.total_memory < 14*1024**3:
        raise RuntimeError(f'GPU memory {props.total_memory/1024**3:.1f} GiB; use a 16 GB-class T4 or larger GPU.')
    return {'name':props.name,'total_memory_bytes':props.total_memory,
            'capability':list(torch.cuda.get_device_capability(0)),
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda}


def runtime():
    packages={}
    for name in ('torch','torchvision','torchaudio','transformers','peft','accelerate','bitsandbytes',
                 'datasets','huggingface-hub','tokenizers','safetensors','numpy','pandas','pyarrow','matplotlib'):
        try: packages[name]=version(name)
        except PackageNotFoundError: packages[name]='not installed'
    return {'python':sys.version,'platform':platform.platform(),'packages':packages}


def load_lock():
    if not LOCK.exists(): raise RuntimeError('Run python -m src.pipeline prepare first.')
    lock=json.loads(LOCK.read_text(encoding='utf-8'))
    if lock['input_sha256']!=data_hashes():
        raise RuntimeError('Data/config/dependency files changed after prepare. Use a separate experiment directory; do not mix runs.')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    return cfg,lock


def get_tokenizer(cfg,lock):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=lock['model_revision'],trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    tok.padding_side='right'
    return tok


def prepare():
    validate_data(ROOT/'artifacts'/'data_validation.json')
    gpu=gpu_required()
    if LOCK.exists():
        cfg,lock=load_lock()
    else:
        from huggingface_hub import HfApi
        cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
        revision=HfApi().model_info(cfg['model_id'],revision='main').sha
        if not revision or len(revision)!=40: raise RuntimeError('Could not resolve a full model commit hash')
        lock={'created_at_utc':now(),'model_id':cfg['model_id'],'model_revision':revision,
              'input_sha256':data_hashes(),'runtime':runtime(),'gpu':gpu,
              'generation':{'do_sample':False,'num_beams':1,'repetition_penalty':1.0,
                            'max_new_tokens':cfg['max_new_tokens']},
              'dataset_type':'synthetic pilot; pending human review'}
        write_json(LOCK,lock)
    tok=get_tokenizer(cfg,lock)
    lengths={}
    for split in ('train','validation','test'):
        rows=read_jsonl(ROOT/'data'/f'{split}.jsonl')
        lens=[len(encode_example(r,tok,cfg['max_length'])['input_ids']) for r in rows]
        lengths[split]={'n':len(lens),'min':min(lens),'max':max(lens),'mean':sum(lens)/len(lens)}
    # Verify actual datasets.load_dataset interoperability, not just json.loads.
    from datasets import load_dataset
    ds=load_dataset('json',data_files={s:str(ROOT/'data'/f'{s}.jsonl') for s in ('train','validation','test')})
    if any(len(ds[s])!=lengths[s]['n'] for s in lengths): raise RuntimeError('Dataset load row mismatch')
    write_json(ROOT/'artifacts'/'tokenization_sanity.json',{'status':'passed','lengths':lengths,
               'assistant_only_labels':True,'datasets_load_dataset_checked':True,'checked_at_utc':now()})
    logs=ROOT/'artifacts'/'logs';logs.mkdir(parents=True,exist_ok=True)
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (logs/'pip_freeze.txt').write_text(freeze,encoding='utf-8')
    try:
        smi=subprocess.check_output(['nvidia-smi'],text=True)
    except (FileNotFoundError,subprocess.CalledProcessError): smi='nvidia-smi unavailable; see CUDA metadata'
    (logs/'nvidia_smi.txt').write_text(smi,encoding='utf-8')
    write_json(logs/'runtime.json',{'runtime':runtime(),'gpu':gpu,'checked_at_utc':now()})
    print(json.dumps({'model_revision':lock['model_revision'],'gpu':gpu,'token_lengths':lengths},ensure_ascii=False,indent=2))


def load_base(cfg,lock):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    gpu_required();set_seed(cfg['seed'])
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModelForCausalLM.from_pretrained(cfg['model_id'],revision=lock['model_revision'],
                    quantization_config=quant,torch_dtype=torch.float16,device_map={'':0},
                    attn_implementation='sdpa',trust_remote_code=False)
    return model


def train(smoke=False):
    import torch
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    cfg,lock=load_lock();gpu_required();set_seed(cfg['seed'])
    kind='smoke' if smoke else 'full'
    adapter_dir=ROOT/'artifacts'/('smoke_adapter' if smoke else 'adapter')
    if (adapter_dir/'adapter_model.safetensors').exists():
        raise FileExistsError(f'{adapter_dir} exists. Preserve the previous run and use a clean project copy.')
    tok=get_tokenizer(cfg,lock)
    encoded={}
    for split in ('train','validation'):
        rows=read_jsonl(ROOT/'data'/f'{split}.jsonl')
        if smoke: rows=rows[:8]
        encoded[split]=Dataset.from_list([encode_example(r,tok,cfg['max_length']) for r in rows])
    model=load_base(cfg,lock)
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                          gradient_checkpointing_kwargs={'use_reentrant':False})
    lora=LoraConfig(r=cfg['lora_r'],lora_alpha=cfg['lora_alpha'],lora_dropout=cfg['lora_dropout'],
                    target_modules=cfg['target_modules'],bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,lora)
    model.print_trainable_parameters()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable: raise RuntimeError('No trainable adapter parameters')
    args=TrainingArguments(
        output_dir=str(ROOT/'runs'/kind),num_train_epochs=cfg['epochs'],max_steps=2 if smoke else -1,
        per_device_train_batch_size=cfg['batch_size'],per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if smoke else cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],lr_scheduler_type='cosine',warmup_ratio=cfg['warmup_ratio'],
        weight_decay=cfg['weight_decay'],max_grad_norm=cfg['max_grad_norm'],
        fp16=True,bf16=False,tf32=False,optim=cfg['optimizer'],
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='no' if smoke else 'epoch',save_strategy='no' if smoke else 'epoch',
        load_best_model_at_end=not smoke,metric_for_best_model='eval_loss',greater_is_better=False,
        save_total_limit=2,save_safetensors=True,logging_steps=1,report_to=[],
        seed=cfg['seed'],data_seed=cfg['seed'],dataloader_num_workers=0,
        remove_unused_columns=False,label_names=['labels'],prediction_loss_only=True,
        eval_accumulation_steps=1)
    trainer=Trainer(model=model,args=args,train_dataset=encoded['train'],eval_dataset=encoded['validation'],
                    data_collator=CompletionCollator(tok.pad_token_id),processing_class=tok)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=trainer.train()
    metrics=dict(result.metrics)
    if not smoke: metrics.update(trainer.evaluate())
    metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   wall_seconds=time.perf_counter()-start,trainable_parameters=trainable)
    adapter_dir.mkdir(parents=True,exist_ok=True)
    trainer.model.save_pretrained(adapter_dir,safe_serialization=True)
    tok.save_pretrained(adapter_dir)
    meta={'training_kind':kind,'model_id':cfg['model_id'],'model_revision':lock['model_revision'],
          'experiment_lock_sha256':sha256(LOCK),'config':cfg,'metrics':metrics,
          'completed_at_utc':now(),'best_model_checkpoint':trainer.state.best_model_checkpoint,
          'runtime':runtime(),'weights_sha256':sha256(adapter_dir/'adapter_model.safetensors')}
    write_json(adapter_dir/'experiment.json',meta)
    write_json(ROOT/'artifacts'/'logs'/f'train_{kind}_history.json',trainer.state.log_history)
    write_json(ROOT/'artifacts'/'logs'/f'train_{kind}_metrics.json',metrics)
    print(json.dumps(meta,ensure_ascii=False,indent=2))


def generate(split='test',adapter=None,out=None,limit=None):
    import copy
    import torch
    from peft import PeftModel
    cfg,lock=load_lock();gpu_required()
    if split not in ('test','validation','demo_prompts'): raise ValueError('Invalid prediction split')
    rows=read_jsonl(ROOT/'data'/f'{split}.jsonl')
    if limit is not None:
        if limit<1: raise ValueError('limit must be positive')
        rows=rows[:limit]
    label='tuned' if adapter else 'baseline'
    out=Path(out) if out else ROOT/'artifacts'/f'{label}_{split}.jsonl'
    if out.exists(): raise FileExistsError(f'{out} exists; preserve it and choose another output path')
    tok=get_tokenizer(cfg,lock);model=load_base(cfg,lock)
    adapter_hash=None
    if adapter:
        adapter=Path(adapter)
        meta=json.loads((adapter/'experiment.json').read_text(encoding='utf-8'))
        if meta['training_kind']!='full': raise ValueError('Smoke adapter is not a trained experiment result')
        if meta['experiment_lock_sha256']!=sha256(LOCK): raise ValueError('Adapter and experiment do not match')
        if meta['weights_sha256']!=sha256(adapter/'adapter_model.safetensors'): raise ValueError('Adapter checksum changed')
        adapter_hash=meta['weights_sha256']
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
    model.eval();model.config.use_cache=True
    gen=copy.deepcopy(model.generation_config)
    gen.do_sample=False;gen.num_beams=1;gen.temperature=1.0;gen.top_p=1.0;gen.top_k=50
    gen.repetition_penalty=1.0;gen.max_new_tokens=cfg['max_new_tokens'];gen.pad_token_id=tok.pad_token_id
    eos=gen.eos_token_id
    eos_set=set(eos if isinstance(eos,list) else [eos])
    out.parent.mkdir(parents=True,exist_ok=True)
    meta={'model_label':label,'model_id':cfg['model_id'],'model_revision':lock['model_revision'],
          'experiment_lock_sha256':sha256(LOCK),'dataset_sha256':sha256(ROOT/'data'/f'{split}.jsonl'),
          'split':split,'expected_ids':[r['id'] for r in rows],'generation':gen.to_dict(),
          'adapter_weights_sha256':adapter_hash,'started_at_utc':now(),'completed':False,'runtime':runtime()}
    write_json(out.with_suffix('.meta.json'),meta)
    with out.open('w',encoding='utf-8') as f:
        for i,row in enumerate(rows,1):
            # Neither reference answer, risk label nor checklist is sent to the model.
            inputs=tok.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True,
                                         return_tensors='pt',return_dict=True).to('cuda:0')
            if inputs['input_ids'].shape[1]>cfg['max_length']: raise ValueError('Prompt too long')
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(): output=model.generate(**inputs,generation_config=gen)
            torch.cuda.synchronize()
            ids=output[0,inputs['input_ids'].shape[1]:].tolist()
            answer=tok.decode(ids,skip_special_tokens=True).strip()
            pred={'id':row['id'],'family_id':row['family_id'],'category':row['category'],
                  'model_label':label,'question':row['messages'][1]['content'],'answer':answer,
                  'answer_sha256':text_sha(answer),'generated_tokens':len(ids),
                  'ended_with_eos':bool(ids and ids[-1] in eos_set),'seconds':time.perf_counter()-start}
            f.write(json.dumps(pred,ensure_ascii=False)+'\n');f.flush()
            print(f'[{i}/{len(rows)}] {row["id"]}: {answer[:180]}',flush=True)
    meta.update(completed=True,completed_at_utc=now(),prediction_sha256=sha256(out),n=len(rows))
    write_json(out.with_suffix('.meta.json'),meta)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('validate');sub.add_parser('prepare')
    t=sub.add_parser('train');t.add_argument('--smoke',action='store_true')
    g=sub.add_parser('generate');g.add_argument('--split',choices=['test','validation','demo_prompts'],default='test')
    g.add_argument('--adapter');g.add_argument('--out');g.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.command=='validate': print(json.dumps(validate_data(ROOT/'artifacts'/'data_validation.json'),ensure_ascii=False,indent=2))
    elif a.command=='prepare':prepare()
    elif a.command=='train':train(a.smoke)
    else:generate(a.split,a.adapter,a.out,a.limit)

if __name__=='__main__':main()
