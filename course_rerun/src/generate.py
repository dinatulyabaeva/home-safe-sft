import argparse,json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from .common import ROOT,read_jsonl,write_jsonl
def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=["validation","bench"],required=True);p.add_argument("--model",choices=["base","sft"],required=True);p.add_argument("--max-new-tokens",type=int,default=512);a=p.parse_args()
 c=json.load(open(ROOT/"configs/experiment.json"));tok=AutoTokenizer.from_pretrained(c["model_id"]);model=AutoModelForCausalLM.from_pretrained(c["model_id"],torch_dtype=torch.float16,device_map="auto")
 if a.model=="sft":model=PeftModel.from_pretrained(model,ROOT/"artifacts/course_adapter")
 model.eval();sys=(ROOT/"configs/system_prompt.txt").read_text().strip()
 rows=read_jsonl(ROOT/("data/home-sft-public-val.jsonl" if a.split=="validation" else "data/home-bench-public.jsonl"));out=[]
 for row in rows:
  question=row["messages"][0]["content"] if a.split=="validation" else row["prompt"][0]["content"]
  msgs=[{"role":"system","content":sys},{"role":"user","content":question}]
  inp=tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(model.device)
  with torch.no_grad():ids=model.generate(inp,max_new_tokens=a.max_new_tokens,do_sample=False,repetition_penalty=1.05,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id)
  ans=tok.decode(ids[0][inp.shape[-1]:],skip_special_tokens=True)
  out.append({"id":row["id"],"qa_id":row["qa_id"],"question":question,"answer":ans,"urgency":row.get("urgency"),"who":row.get("who"),"category":row.get("category"),"difficulty":row.get("difficulty"),"rubric":row.get("rubric") if a.split=="bench" else None})
 write_jsonl(ROOT/"artifacts"/f"{a.split}_{a.model}.jsonl",out);print("Saved",len(out))
if __name__=="__main__":main()
