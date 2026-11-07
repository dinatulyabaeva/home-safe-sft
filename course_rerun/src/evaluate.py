import json,statistics
from .common import ROOT,read_jsonl,parse_answer
def score(preds,refs):
 gold={r["id"]:r for r in refs};fmt=[];u=[];w=[];c=[];d=[]
 for p in preds:
  r=gold[p["id"]];x=parse_answer(p["answer"]);lab=x["labels"];fmt.append(x["format_valid"])
  u.append(lab.get("urgency")==r["urgency"]);w.append(lab.get("who")==r["who"]);c.append(lab.get("category")==r["category"])
  try:d.append(abs(int(lab.get("difficulty","").split("/")[0])-r["difficulty"]))
  except:d.append(9)
 return {"n":len(preds),"format_valid_rate":sum(fmt)/len(fmt),"urgency_accuracy":sum(u)/len(u),"who_accuracy":sum(w)/len(w),"category_accuracy":sum(c)/len(c),"difficulty_mae":statistics.mean(d)}
def main():
 refs=read_jsonl(ROOT/"data/home-sft-public-val.jsonl");res={}
 for m in ("base","sft"):
  p=ROOT/"artifacts"/f"validation_{m}.jsonl"
  if p.exists():res[m]=score(read_jsonl(p),refs)
 if len(res)==2:
  json.dump(res,open(ROOT/"artifacts/course_val_metrics.json","w"),ensure_ascii=False,indent=2);print(json.dumps(res,ensure_ascii=False,indent=2))
 else:print("Generate validation_base.jsonl and validation_sft.jsonl first")
if __name__=="__main__":main()
