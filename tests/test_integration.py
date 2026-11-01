"""End-to-end CPU tests use artificial responses only in a temporary directory.
They do NOT create or claim real baseline/tuned experiment artifacts.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.common import ROOT,read_jsonl,write_json,write_jsonl,sha256,text_sha
import src.evaluate as ev
import src.judge as judge

class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for d in ('data','artifacts','reports','configs'):(self.root/d).mkdir()
        self.data=read_jsonl(ROOT/'data'/'test.jsonl')[:3]
        write_jsonl(self.root/'data'/'test.jsonl',self.data)
        (self.root/'configs'/'judge_system.txt').write_text((ROOT/'configs'/'judge_system.txt').read_text(encoding='utf-8'),encoding='utf-8')
        for label in ('baseline','tuned'):
            rows=[]
            for r in self.data:
                answer='[ARTIFICIAL TEST FIXTURE] '+r['messages'][2]['content']
                rows.append({'id':r['id'],'family_id':r['family_id'],'category':r['category'],
                  'model_label':label,'question':r['messages'][1]['content'],'answer':answer,
                  'answer_sha256':text_sha(answer),'generated_tokens':100,'ended_with_eos':True,'seconds':0})
            path=self.root/'artifacts'/f'{label}_test.jsonl';write_jsonl(path,rows)
            write_json(path.with_suffix('.meta.json'),{'completed':True,'model_label':label,
                'prediction_sha256':sha256(path),'dataset_sha256':sha256(self.root/'data'/'test.jsonl'),
                'model_id':'ARTIFICIAL_TEST_MODEL','model_revision':'TEST_ONLY','experiment_lock_sha256':'TEST_ONLY',
                'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.,'eos_token_id':2,'pad_token_id':0}})
        self.p1=patch.object(ev,'ROOT',self.root);self.p1.start()
        self.p2=patch.object(judge,'ROOT',self.root);self.p2.start()
        self.p3=patch.object(judge,'OUT',self.root/'artifacts'/'judge');self.p3.start()
    def tearDown(self):
        self.p3.stop();self.p2.stop();self.p1.stop();self.tmp.cleanup()
    def test_evaluate_and_judge_export_import(self):
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(self.root/'artifacts'/'baseline_test.jsonl',self.root/'artifacts'/'tuned_test.jsonl')
            judge.export_tasks()
        tasks=read_jsonl(judge.OUT/'tasks.jsonl')
        self.assertEqual(len(tasks),6)
        for task in tasks:
            self.assertNotIn('model_label',json.dumps(task))
        results=[{'task_id':t['task_id'],'safety':4,'correctness':4,'helpfulness':4,
                  'critical_error':False,'unnecessary_refusal':False,'reason':'ARTIFICIAL TEST FIXTURE'} for t in tasks]
        write_jsonl(judge.OUT/'results.jsonl',results)
        with contextlib.redirect_stdout(io.StringIO()):judge.summarize()
        summary=json.loads((judge.OUT/'summary.json').read_text())
        self.assertTrue(summary['complete']);self.assertEqual(summary['n_complete_pairs'],3)
    def test_partial_predictions_rejected(self):
        path=self.root/'artifacts'/'baseline_test.jsonl'
        write_jsonl(path,read_jsonl(path)[:-1])
        meta=json.loads(path.with_suffix('.meta.json').read_text());meta['prediction_sha256']=sha256(path)
        write_json(path.with_suffix('.meta.json'),meta)
        with self.assertRaises(ValueError):ev.verified_pair(path,self.root/'artifacts'/'tuned_test.jsonl')
    def test_changed_decoding_rejected(self):
        path=self.root/'artifacts'/'tuned_test.meta.json'
        m=json.loads(path.read_text());m['generation']['max_new_tokens']=256;write_json(path,m)
        with self.assertRaises(ValueError):ev.verified_pair(self.root/'artifacts'/'baseline_test.jsonl',self.root/'artifacts'/'tuned_test.jsonl')
    def test_missing_judge_results_not_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):judge.export_tasks()
        t=read_jsonl(judge.OUT/'tasks.jsonl')[0]
        r={'task_id':t['task_id'],'safety':4,'correctness':4,'helpfulness':4,
           'critical_error':False,'unnecessary_refusal':False,'reason':'ARTIFICIAL TEST FIXTURE'}
        write_jsonl(judge.OUT/'results.jsonl',[r])
        with contextlib.redirect_stdout(io.StringIO()):judge.summarize()
        s=json.loads((judge.OUT/'summary.json').read_text());self.assertEqual(s['n_complete_pairs'],0)
        self.assertIsNone(s['metrics_complete_pairs_only'][0]['delta'])

if __name__=='__main__':unittest.main()
