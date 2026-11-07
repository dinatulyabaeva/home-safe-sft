import json,re,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read_jsonl(p):
 with open(p,encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def write_jsonl(p,rows):
 pathlib.Path(p).parent.mkdir(parents=True,exist_ok=True)
 with open(p,"w",encoding="utf-8") as f:
  for x in rows:f.write(json.dumps(x,ensure_ascii=False)+"\n")
def parse_answer(text):
 lines=[x.strip() for x in text.strip().splitlines() if x.strip()]
 keys=("Срочность:","Кто:","Категория:","Сложность:")
 idx=next((i for i in range(len(lines)-4,-1,-1) if all(lines[i+j].startswith(keys[j]) for j in range(4))),None)
 labels={}
 if idx is not None:
  for j,k in enumerate(("urgency","who","category","difficulty")):labels[k]=lines[idx+j].split(":",1)[1].strip()
 steps=[int(x) for x in re.findall(r"(?m)^\s*(\d+)\.\s+",text)]
 words=len(re.findall(r"\b[\wА-Яа-яЁё-]+\b",text))
 intro=text.split("Что сделать:",1)[0].strip() if "Что сделать:" in text else ""
 intro_n=len(re.findall(r"[.!?]+",intro))
 valid=(idx is not None and idx+4==len(lines) and 2<=len(steps)<=5 and steps==list(range(1,len(steps)+1)) and 2<=intro_n<=4 and words<=150 and not re.search(r"(?m)^#{1,6}\s",text))
 return {"labels":labels,"steps":steps,"word_count":words,"intro_sentence_count":intro_n,"four_labels_at_end":idx is not None and idx+4==len(lines),"format_valid":valid}
