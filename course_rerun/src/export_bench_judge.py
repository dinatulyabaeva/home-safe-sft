from .common import ROOT,read_jsonl,write_jsonl
def main():
 tasks=[]
 for model in ("base","sft"):
  rows=read_jsonl(ROOT/f"artifacts/bench_{model}.jsonl")
  for r in rows:
   tasks.append({"task_id":r["id"]+"-"+model,"bench_id":r["id"],"model":model,"question":r["question"],"answer_to_evaluate":r["answer"],"rubric":r["rubric"],"instruction":"Оцени каждый критерий рубрики независимо. Не угадывай модель. Для каждого критерия верни criterion_index (индекс с нуля), met=true/false, reason. Верни только JSON: {task_id, scores:[{criterion_index,met,reason}]}."})
 write_jsonl(ROOT/"artifacts/bench_judge_tasks.jsonl",tasks);print("Exported",len(tasks),"rubric tasks (30 per model)")
if __name__=="__main__":main()
