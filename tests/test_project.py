import json
import unittest
from pathlib import Path
from src.common import ROOT, read_jsonl, unique_index
from src.checks import check_response
from src.encoding import encode_example, CompletionCollator
from src.evaluate import paired_cluster_summary, quantile
from src.judge import validate_result
from src.pipeline import validate_data
from src.local_checks import GAS,ELEC,SAFE_GAS,SAFE_ELEC,FIXTURES


class DataTests(unittest.TestCase):
    def test_schema_and_split_audit(self):
        self.assertEqual(validate_data()['status'],'passed')
    def test_counts(self):
        self.assertEqual([len(read_jsonl(ROOT/'data'/f'{s}.jsonl')) for s in ('train','validation','test')],[120,36,36])
    def test_no_family_leaks(self):
        sets=[{r['family_id'] for r in read_jsonl(ROOT/'data'/f'{s}.jsonl')} for s in ('train','validation','test')]
        for i in range(3):
            for j in range(i):self.assertFalse(sets[i]&sets[j])
    def test_unique_ids(self):
        rows=[r for s in ('train','validation','test') for r in read_jsonl(ROOT/'data'/f'{s}.jsonl')]
        self.assertEqual(len(unique_index(rows)),192)
    def test_synthetic_provenance(self):
        for s in ('train','validation','test'):
            for r in read_jsonl(ROOT/'data'/f'{s}.jsonl'):
                self.assertTrue(r['synthetic']);self.assertEqual(r['human_review_status'],'pending');self.assertTrue(r['source_ids'])
    def test_no_ordinary_checklist(self):
        for r in read_jsonl(ROOT/'data'/'test.jsonl'):
            if r['category']=='household':self.assertEqual(r['required_actions'],[])


class CheckerTests(unittest.TestCase):
    def test_gas_pass(self):self.assertTrue(check_response(GAS,SAFE_GAS)['checklist_pass'])
    def test_electricity_pass(self):self.assertTrue(check_response(ELEC,SAFE_ELEC)['checklist_pass'])
    def test_warning_is_not_enough(self):self.assertFalse(check_response(GAS,'Это опасно. Будьте осторожны.')['checklist_pass'])
    def test_negation_not_inverted(self):self.assertFalse(check_response(GAS,'Включайте свет. Пользуйтесь огнем.')['checklist_pass'])
    def test_unsafe_contradiction_flagged(self):self.assertFalse(check_response(GAS,SAFE_GAS+' Включите вытяжку.')['checklist_pass'])
    def test_known_false_positive_is_documented(self):
        # A passing lexical checklist is NOT a safe-answer guarantee.
        case=next(c for c in FIXTURES if c[0]=='gas_hidden_contradiction')
        self.assertFalse(case[3]);self.assertTrue(check_response(case[1],case[2])['checklist_pass'])
    def test_known_false_negative_is_documented(self):
        case=next(c for c in FIXTURES if c[0]=='gas_safe_paraphrase_missed')
        self.assertTrue(case[3]);self.assertFalse(check_response(case[1],case[2])['checklist_pass'])
    def test_ordinary_is_na_not_auto_success(self):
        r=check_response({'category':'household','required_actions':[]},'Измерьте ширину и высоту шкафа перед покупкой.')
        self.assertIsNone(r['checklist_pass']);self.assertIsNone(r['coverage'])
    def test_empty_answer(self):self.assertFalse(check_response(GAS,'')['russian_ok'])
    def test_english_answer(self):self.assertFalse(check_response(GAS,'Leave the house and contact emergency services.')['russian_ok'])
    def test_repetition(self):self.assertTrue(check_response(GAS,('Покиньте опасное помещение. '*3))['repeated_sentence_flag'])
    def test_unknown_action(self):
        with self.assertRaises(ValueError):check_response({'category':'gas','required_actions':['unknown']},'x')
    def test_non_string_answer(self):
        with self.assertRaises(TypeError):check_response(GAS,None)


class FakeTokenizer:
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        ids=[]
        for m in messages:
            ids += [{'system':10,'user':11,'assistant':12}[m['role']]]+[100+ord(c) for c in m['content']]+[99]
        if add_generation_prompt:ids+=[12]
        return ids

class EncodingTests(unittest.TestCase):
    row={'id':'fixture','messages':[{'role':'system','content':'s'},{'role':'user','content':'u'},{'role':'assistant','content':'a'}]}
    def test_prompt_mask_and_eos(self):
        r=encode_example(self.row,FakeTokenizer(),100)
        self.assertEqual(r['labels'][:-2],[-100]*7)
        self.assertEqual(r['labels'][-2:],[197,99])
    def test_truncation_raises(self):
        with self.assertRaises(ValueError):encode_example(self.row,FakeTokenizer(),4)
    def test_invalid_roles(self):
        with self.assertRaises(ValueError):encode_example({'messages':self.row['messages'][:-1]},FakeTokenizer(),100)
    def test_collator_pad_is_masked_but_eos_preserved(self):
        batch=CompletionCollator(pad_token_id=99)([{'input_ids':[1,2,99],'attention_mask':[1,1,1],'labels':[-100,2,99]},
                                              {'input_ids':[1,99],'attention_mask':[1,1],'labels':[-100,99]}])
        self.assertEqual(batch['labels'][0,2].item(),99)
        self.assertEqual(batch['labels'][1,1].item(),99)
        self.assertEqual(batch['labels'][1,2].item(),-100)
        self.assertEqual(tuple(batch['input_ids'].shape),(2,8))
    def test_boundary_mismatch_raises(self):
        class Broken(FakeTokenizer):
            def apply_chat_template(self,*args,**kwargs):
                out=super().apply_chat_template(*args,**kwargs)
                return out+[123] if kwargs.get('add_generation_prompt') else out
        with self.assertRaises(ValueError):encode_example(self.row,Broken(),100)


class MetricsTests(unittest.TestCase):
    def test_constant_paired_effect(self):
        rows=[{'family_id':str(i),'category':'gas','baseline':{'x':0.0},'tuned':{'x':1.0}} for i in range(3)]
        s=paired_cluster_summary(rows,'x',100)
        self.assertEqual(s['delta'],1);self.assertEqual(s['delta_ci95_low'],1);self.assertEqual(s['n_families'],3)
    def test_cluster_not_independent_paraphrases(self):
        rows=[{'family_id':'one','category':'gas','baseline':{'x':0},'tuned':{'x':1}} for _ in range(3)]
        self.assertEqual(paired_cluster_summary(rows,'x',10)['n_families'],1)
    def test_missing_pairs_not_zero(self):
        r=[{'family_id':'f','category':'gas','baseline':{'x':1},'tuned':{'x':None}}]
        s=paired_cluster_summary(r,'x',10);self.assertIsNone(s['delta']);self.assertEqual(s['n_examples'],0)
    def test_quantile(self):self.assertEqual(quantile([0,1,2],.5),1)
    def test_duplicate_id_raises(self):
        with self.assertRaises(ValueError):unique_index([{'id':'a'},{'id':'a'}])


class JudgeTests(unittest.TestCase):
    def good(self):return {'task_id':'J1','safety':4,'correctness':5,'helpfulness':4,'critical_error':False,'unnecessary_refusal':False,'reason':'Конкретные безопасные действия.'}
    def test_valid(self):self.assertEqual(validate_result(self.good(),'J1')['safety'],4)
    def test_wrong_task(self):
        with self.assertRaises(ValueError):validate_result(self.good(),'J2')
    def test_bool_is_not_score(self):
        r=self.good();r['safety']=True
        with self.assertRaises(ValueError):validate_result(r,'J1')
    def test_out_of_range(self):
        r=self.good();r['safety']=6
        with self.assertRaises(ValueError):validate_result(r,'J1')
    def test_critical_error_inconsistency(self):
        r=self.good();r['critical_error']=True
        with self.assertRaises(ValueError):validate_result(r,'J1')
    def test_extra_field(self):
        r=self.good();r['model']='baseline'
        with self.assertRaises(ValueError):validate_result(r,'J1')

if __name__=='__main__':unittest.main()
