from preanchor.labels.convergence import convergence_idx, eligible
from preanchor.labels.threshold import train_threshold

def test_convergence_and_exclusion():
    assert convergence_idx([.2,.99,.995,.999,.99,.995,.1]) == 1
    assert eligible(0,1) and not eligible(1,1)

def test_threshold_training_only():
    rows=[{'problem_id':1,'importance_primary':1},{'problem_id':1,'importance_primary':2},{'problem_id':2,'importance_primary':100}]
    assert train_threshold(rows,{1},.9)==1.9
