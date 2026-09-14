"""Leakage-safe training-only anchor thresholds."""
def train_threshold(rows: list[dict], train_problem_ids: set[int], quantile: float=.90) -> float:
    if not 0 < quantile < 1: raise ValueError('quantile must be in (0,1)')
    values=[float(r['importance_primary']) for r in rows if r.get('problem_id') in train_problem_ids and r.get('eligible',True)]
    if not values: raise ValueError('no eligible training labels')
    values.sort(); pos=(len(values)-1)*quantile; lo=int(pos); hi=min(lo+1,len(values)-1); return values[lo]+(values[hi]-values[lo])*(pos-lo)
