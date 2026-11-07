import json
from collections import defaultdict
from .common import ROOT,read_jsonl
def main():
 tasks=read_jsonl(ROOT/"artifacts/bench_judge_tasks.jsonl");results=read_jsonl(ROOT/"artifacts/bench_judge_results.jsonl")
 taskmap={x["task_id"]:x for x in tasks};seen=set();grouped=defaultdict(list)
 for r in results:
  tid=r["task_id"]
  if tid not in taskmap or tid in seen:raise ValueError("Unknown or duplicate task_id: "+tid)
  seen.add(tid);rubric=taskmap[tid]["rubric"];scores=r["scores"]
  if len(scores)!=len(rubric):raise ValueError("Rubric length mismatch: "+tid)
  total=0;indices=set()
  for sc in scores:
   i=sc["criterion_index"]
   if type(i)is not int or i<0 or i>=len(rubric) or i in indices:raise ValueError("Invalid criterion index: "+tid)
   indices.add(i)
   if type(sc["met"])is not bool:raise ValueError("met must be boolean")
   if sc["met"]:total+=rubric[i]["points"]
  grouped[taskmap[tid]["model"]].append(total)
 if len(seen)!=len(taskmap):raise ValueError(f"Missing {len(taskmap)-len(seen)} tasks; missing results are not zero")
 summary={m:{"n":len(v),"mean_weighted_rubric_points":sum(v)/len(v),"min":min(v),"max":max(v)} for m,v in grouped.items()}
 json.dump(summary,open(ROOT/"artifacts/bench_rubric_summary.json","w"),ensure_ascii=False,indent=2);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
