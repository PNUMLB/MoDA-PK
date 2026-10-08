import sys,unittest
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pkforecast.metrics import metric
class MetricTests(unittest.TestCase):
 def test_hierarchy_not_pooled_or_mean_person_rmse(self):
  # Person 1 has errors 0,0,0; person 2 has error4: hierarchical sqrt8,
  # not pooled2 nor mean-person RMSE2. A second drug error1 has equal drug weight.
  p=pd.DataFrame([['x',1,'s','d1','p1',0]]*3+[['x',1,'s','d1','p2',16],['x',1,'s','d2','p3',1]],columns=['condition','seed','source_id','drug_id','participant_id','sq'])
  _,s=metric(p);self.assertAlmostEqual(s.sb_hrmse.iloc[0],(np.sqrt(8)+1)/2)
 def test_seed_scores_not_ensemble(self):
  p=pd.DataFrame([['x',1,'s','d','p',4],['x',2,'s','d','p',4]],columns=['condition','seed','source_id','drug_id','participant_id','sq']);_,s=metric(p)
  self.assertEqual(len(s),2);self.assertEqual(s.sb_hrmse.mean(),2.)
if __name__=='__main__':unittest.main()
