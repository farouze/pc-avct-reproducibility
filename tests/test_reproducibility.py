import ast
import json
from pathlib import Path
import re
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from inference import holm_adjust, sign_flip_pvalue

def helpers():
    namespace=dict(np=np,pd=pd,re=re,SEED=42,WINDOW_SECONDS=10,V26_CAL_PATTERN=r'^\s*calibration\s+start',StratifiedKFold=StratifiedKFold)
    exec(compile((ROOT/'analysis/original_v26_functions.py').read_text(),'v26_helpers','exec'),namespace)
    return namespace

class ReproducibilityTests(unittest.TestCase):
    def test_source_matches_supplied_lean_notebook(self):
        book=json.loads((ROOT/'notebooks/AVCT_Aurora_LEAN_V26_V28_V29_V30.ipynb').read_text())
        funcs={}
        for cell in book['cells']:
            if cell['cell_type']!='code':continue
            src='\n'.join('pass' if line.startswith(('!','%')) else line for line in ''.join(cell['source']).splitlines())
            for node in ast.parse(src).body:
                if isinstance(node,ast.FunctionDef):funcs[node.name]=ast.dump(node)
        nodes=ast.parse((ROOT/'analysis/original_v26_functions.py').read_text()).body
        self.assertEqual(len(nodes),14)
        for node in nodes:self.assertEqual(ast.dump(node),funcs[node.name],node.name)

    def test_calibration_dose_never_scores_initial_block(self):
        # The one-reading arm must still start after the entire three-reading block.
        h=helpers();rows=[]
        for i,(label,sbp,feature) in enumerate([('Calibration start 1',120,1),('Calibration start 2',126,3),('Calibration start 3',132,5),('Seated',140,9)]):
            rows.append(dict(pid='synthetic',phase='initial',measurement=label,date_time=f'2020-01-01 00:00:0{i}',_source_order=i,sbp=sbp,dbp=80+i,raw_sigma_m=feature))
        frame=pd.DataFrame(rows)
        one,_=h['v26_build_calibration_dose'](frame,1);three,_=h['v26_build_calibration_dose'](frame,3)
        self.assertEqual(len(one),2);self.assertEqual(len(three),2)
        self.assertEqual(one.iloc[0].sbp,120);self.assertEqual(three.iloc[0].sbp,126)
        self.assertEqual(three.iloc[0].raw_sigma_m,3)
        self.assertEqual(one.iloc[1].start_seconds,1);self.assertEqual(three.iloc[1].start_seconds,1)
        delta=h['build_delta_table'](three,['raw_sigma_m'],[],anchor='session_start')
        self.assertEqual(list(delta.measurement),['Seated']);self.assertEqual(delta.iloc[0].delta_sbp,14)

    def test_outer_folds_keep_all_visits_of_each_participant_together(self):
        h=helpers();f=pd.DataFrame([dict(subject_id=f's{p:02}',fst_band='a' if p<10 else 'b',visit=v) for p in range(20) for v in range(3)])
        seen=[]
        for _,train,test in h['v26_participant_folds'](f,5,0):
            self.assertFalse(set(train.subject_id)&set(test.subject_id))
            self.assertTrue(test.groupby('subject_id').size().eq(3).all());seen.extend(test.subject_id.unique())
        self.assertEqual(len(seen),20);self.assertEqual(len(set(seen)),20)

    def test_holm_corrects_in_original_endpoint_order(self):
        np.testing.assert_allclose(holm_adjust([.04,.01,.03]),[.06,.03,.06])
        np.testing.assert_allclose(holm_adjust([.00001,.00013]),[.00002,.00013])

    def test_paired_test_null_and_sign_symmetry(self):
        self.assertEqual(sign_flip_pvalue([0,0,0],999),1)
        d=np.arange(1,21,dtype=float)
        self.assertEqual(sign_flip_pvalue(d,9999),sign_flip_pvalue(-d,9999))
        self.assertLess(sign_flip_pvalue(d,9999),.01)
        with self.assertRaises(ValueError):sign_flip_pvalue([1,np.nan])

if __name__=='__main__':unittest.main()
