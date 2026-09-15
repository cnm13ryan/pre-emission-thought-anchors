import json,time
from pathlib import Path
from preanchor.labels.importance import canonical_from_forks
root=Path('/lambda/nfs/preanchor/preanchor/artifacts/runs/20260914T183017Z-1c085746e3-08bc1e7c/stages/stage2/units')
paths=list(root.glob('*/solutions.json'))
paths=paths[:10]
t=time.monotonic(); n=0
for p in paths:
 d=json.loads(p.read_text()); canonical_from_forks({0:d,1:d},0); n+=len(d) if isinstance(d,list) else 0
print(json.dumps({'files':len(paths),'samples':n,'seconds':time.monotonic()-t,'units_per_second':len(paths)/(time.monotonic()-t)}))
