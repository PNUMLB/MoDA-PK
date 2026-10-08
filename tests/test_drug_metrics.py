import numpy as np
from mhap_pk.evaluation.drug_metrics import balanced_metrics

def test_equal_participant_weight_and_r2():
    m=balanced_metrics([0.,2.,4.],[1.,3.,4.],['a','a','b'])
    assert np.isclose(m['log_rmse'],np.sqrt(.5))
    assert np.isclose(m['r2_log'],1-.5/2.75)
    assert np.isclose(m['linear_rmse'],np.sqrt(.25*(np.e-1)**2+.25*(np.exp(3)-np.exp(2))**2))
def test_undefined_and_negative_r2():
    assert np.isnan(balanced_metrics([1],[2],['a'])['r2_log'])
    assert np.isnan(balanced_metrics([1,1],[2,2],['a','b'])['r2_linear'])
    assert balanced_metrics([0,1],[10,11],['a','b'])['r2_log']<0

def test_repeat_measurements_does_not_reweight_subject():
    a=balanced_metrics([0,2],[1,2],['a','b'])
    b=balanced_metrics([0,0,2],[1,1,2],['a','a','b'])
    assert np.isclose(a['log_rmse'],b['log_rmse'])
