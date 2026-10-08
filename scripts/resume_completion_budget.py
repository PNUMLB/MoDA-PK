"""Resume scheduling only; existing live fit subprocesses are never interrupted."""
from pathlib import Path
import sys,json,subprocess,os,time,signal
R=Path(__file__).resolve().parents[1];B=R/'results/runs/COMPLETION_SUITE/20261002'
plan=json.loads((B/'registered_plan.json').read_text());jobs=plan['budget_jobs'];children={};start=time.time()
def save(p,x):tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def state(i):
 j=jobs[i];d=B/'budget240'/f"{j['fold']}__s{j['seed']}__{j['mode']}";f=d/'run_manifest.json'
 return json.loads(f.read_text()).get('status','running') if f.exists() else ('running' if d.exists() else 'pending')
# Stop only the old scheduling parent. Its model-fit subprocesses continue unchanged.
lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines();parents=[]
for line in lines:
 parts=line.strip().split(None,1)
 if len(parts)==2 and parts[1] in ['python scripts/run_completion_suite.py',sys.executable+' scripts/run_completion_suite.py']:
  pid=int(parts[0]);os.kill(pid,signal.SIGTERM);parents.append(pid)
save(B/'scheduler_resume.json',dict(parent_schedulers_stopped=parents,fit_workers_interrupted=False,reason='Use free CPU slots as short controls complete; identical registered fit jobs',maximum_total_fit_processes=56))
while True:
 statuses=[('running' if i in children and children[i][0].poll() is None else state(i)) for i in range(len(jobs))]
 for i,(p,f) in list(children.items()):
  if p.poll() is not None:
   f.close()
   if p.returncode and statuses[i]=='pending':raise RuntimeError(f'Worker {i} exited without manifest')
   del children[i]
 active=sum(s=='running' for s in statuses);done=sum(s=='completed' for s in statuses);failed=sum(s=='failed' for s in statuses)
 cp=B/'controls_plan.json';control_active=0
 if cp.exists():
  for j in json.loads(cp.read_text())['jobs']:
   f=B/j['variant']/f"{j['fold']}__s{j['seed']}__molecular_relation"/'run_manifest.json'
   if f.exists() and json.loads(f.read_text()).get('status')=='running':control_active+=1
 cap=max(24,56-control_active)
 pending=[i for i,s in enumerate(statuses) if s=='pending']
 mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());available_gib=int(mem['MemAvailable'].split()[0])/1024**2
 memory_slots=max(0,int((available_gib-80)/6))
 for i in pending[:min(max(0,cap-active),memory_slots)]:
  f=(B/'logs'/f'budget_{i:03d}.log').open('a');p=subprocess.Popen([sys.executable,str(R/'scripts/run_completion_suite.py'),'--job',str(i)],cwd=R,stdout=f,stderr=subprocess.STDOUT,env=os.environ|dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));children[i]=(p,f);active+=1
 save(B/'budget_status.json',dict(status='completed' if done==len(jobs) else ('failed' if done+failed==len(jobs) else 'running'),completed=done,failed=failed,total=len(jobs),active=active,worker_cap=cap,available_memory_gib=available_gib,elapsed_since_resume=time.time()-start,jobs=[j|dict(status=s) for j,s in zip(jobs,statuses)]));print(done,failed,active,'cap',cap,flush=True)
 if done+failed==len(jobs):break
 time.sleep(10)
