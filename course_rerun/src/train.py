import json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,Trainer,TrainingArguments,DataCollatorForSeq2Seq,set_seed
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset
from .common import ROOT
def main():
 if not torch.cuda.is_available():
  raise RuntimeError("PyTorch не видит GPU. Выберите T4 GPU в Colab.")
 c=json.load(open(ROOT/"configs/experiment.json"));set_seed(c["seed"]);mid=c["model_id"]
 tok=AutoTokenizer.from_pretrained(mid);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side="right"
 sys=(ROOT/"configs/system_prompt.txt").read_text().strip()
 raw=load_dataset("json",data_files={"train":str(ROOT/"data/home-sft-public-train.jsonl"),"validation":str(ROOT/"data/home-sft-public-val.jsonl")})
 class DS(torch.utils.data.Dataset):
  def __init__(self,s):self.rows=raw[s]
  def __len__(self):return len(self.rows)
  def __getitem__(self,i):
   row=self.rows[i]; msgs=[{"role":"system","content":sys}]+row["messages"]
   prompt=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True)
   full=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)

   # Явное сокращение только слишком длинного вопроса
   if len(full)>c["max_seq_length"]:
    original_user=msgs[1]["content"]
    marker="\\n[Средняя часть длинного сообщения опущена при подготовке примера]\\n"
    lo,hi=0,len(original_user)
    best=None

    while lo<=hi:
     keep=(lo+hi)//2
     head_n=(keep*3)//5
     tail_n=keep-head_n

     if keep==0:
      candidate=marker.strip()
     else:
      candidate=original_user[:head_n]+marker+(original_user[-tail_n:] if tail_n else "")

     msgs[1]["content"]=candidate
     p_ids=tok.apply_chat_template(msgs[:-1],tokenize=True,add_generation_prompt=True)
     f_ids=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False)

     if len(f_ids)<=c["max_seq_length"] and len(p_ids)<len(f_ids):
      best=(candidate,p_ids,f_ids,keep)
      lo=keep+1
     else:
      hi=keep-1

    if best is None:
     raise ValueError(f"Даже без текста вопроса ответ превышает лимит: {row['id']}")

    msgs[1]["content"],prompt,full,kept_chars=best
    print(f"[TRUNCATED INPUT] {row['id']}: {len(original_user)} -> {kept_chars} символов. Ответ сохранён полностью.")

   if len(prompt)>=len(full) or len(full)>c["max_seq_length"]:
    raise ValueError(f"Некорректная длина: {row['id']}")

   labels=full.copy()
   labels[:len(prompt)]=[-100]*len(prompt)
   return {"input_ids":full,"attention_mask":[1]*len(full),"labels":labels}
 q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
 model=AutoModelForCausalLM.from_pretrained(mid,quantization_config=q,device_map="auto",dtype=torch.float16)
 model.config.use_cache=False;model=prepare_model_for_kbit_training(model)
 model=get_peft_model(model,LoraConfig(r=c["lora_r"],lora_alpha=c["lora_alpha"],lora_dropout=c["lora_dropout"],bias="none",task_type="CAUSAL_LM",target_modules=c["target_modules"]))
 args=TrainingArguments(output_dir=str(ROOT/"artifacts/course_training"),num_train_epochs=c["epochs"],per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,learning_rate=c["learning_rate"],fp16=True,eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,gradient_checkpointing=True,optim="paged_adamw_8bit",warmup_ratio=.1,weight_decay=.01,logging_steps=1,report_to="none")
 trainer=Trainer(model=model,args=args,train_dataset=DS("train"),eval_dataset=DS("validation"),data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100))
 trainer.train();out=ROOT/"artifacts/course_adapter";out.mkdir(parents=True,exist_ok=True);trainer.model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
 json.dump(trainer.state.log_history,open(ROOT/"artifacts/course_training_metrics.json","w"),ensure_ascii=False,indent=2)
 print("Saved adapter:",out)
if __name__=="__main__":main()
