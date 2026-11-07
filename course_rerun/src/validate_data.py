from datasets import load_dataset
from collections import Counter
from .common import ROOT,read_jsonl
def main():
 tr=read_jsonl(ROOT/"data/home-sft-public-train.jsonl"); va=read_jsonl(ROOT/"data/home-sft-public-val.jsonl")
 be=read_jsonl(ROOT/"data/home-bench-public.jsonl"); qa=read_jsonl(ROOT/"data/home-qa-public.jsonl")
 assert (len(qa),len(tr),len(va),len(be))==(100,50,20,30)
 ids=[x["id"] for x in tr+va+be]; assert len(ids)==len(set(ids))
 qids=[x["qa_id"] for x in tr+va+be]; assert len(qids)==len(set(qids)),"SFT/val/bench overlap"
 for x in tr+va:
  assert [m["role"] for m in x["messages"]]==["user","assistant"]
  assert x["urgency"] in ("планово","скоро","срочно") and x["who"] in ("сам","специалист")
  assert isinstance(x["difficulty"],int) and 1<=x["difficulty"]<=10
 for x in be: assert x["prompt"] and isinstance(x["rubric"],list) and all(-10<=c["points"]<=10 for c in x["rubric"])
 ds=load_dataset("json",data_files={"train":str(ROOT/"data/home-sft-public-train.jsonl"),"validation":str(ROOT/"data/home-sft-public-val.jsonl")})
 print({"qa":len(qa),"train":len(tr),"validation":len(va),"bench":len(be),"train_categories":dict(Counter(x["category"] for x in tr)),"rubric_criteria":sum(len(x["rubric"]) for x in be),"datasets_load_dataset":{k:len(v) for k,v in ds.items()},"qa_id_overlap":0})
if __name__=="__main__":main()
