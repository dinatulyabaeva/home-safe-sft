"""Completion-only encoding; never truncate away the supervised assistant answer."""
from __future__ import annotations
from dataclasses import dataclass


def encode_example(row: dict, tokenizer, max_length: int) -> dict:
    messages = row['messages']
    if [m['role'] for m in messages] != ['system','user','assistant']:
        raise ValueError(f"{row.get('id')}: expected one system/user/assistant exchange")
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"{row.get('id')}: chat template boundary mismatch; do not guess the assistant mask")
    if len(full)>max_length:
        raise ValueError(f"{row.get('id')}: {len(full)} tokens > max_length={max_length}; revise data/config, not silent truncation")
    if len(full)<=len(prompt):
        raise ValueError(f"{row.get('id')}: no assistant tokens")
    labels = [-100]*len(prompt) + full[len(prompt):]
    return {'input_ids':full, 'attention_mask':[1]*len(full), 'labels':labels}

@dataclass
class CompletionCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict]) -> dict:
        import torch
        if not features:
            raise ValueError('empty batch')
        size = max(len(f['input_ids']) for f in features)
        m = self.pad_to_multiple_of
        size = ((size+m-1)//m)*m
        batch = {'input_ids':[], 'attention_mask':[], 'labels':[]}
        for f in features:
            length=len(f['input_ids']); pad=size-length
            if not (length==len(f['attention_mask'])==len(f['labels'])):
                raise ValueError('inconsistent sequence lengths')
            batch['input_ids'].append(f['input_ids']+[self.pad_token_id]*pad)
            batch['attention_mask'].append(f['attention_mask']+[0]*pad)
            batch['labels'].append(f['labels']+[-100]*pad)
        return {k:torch.tensor(v,dtype=torch.long) for k,v in batch.items()}
