from pathlib import Path
import json,hashlib,torch
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str))
def inputs(b,k):return b.inputs(torch.full((len(b.indices),),k,dtype=torch.long))
