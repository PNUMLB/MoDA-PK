"""Finish aggregation and paper exports only after every registered run succeeds."""
from pathlib import Path
import json,time,subprocess,sys,os,shutil
R=Path(__file__).resolve().parents[1];B=R/'results/runs/COMPLETION_SUITE/20261002';M=R/'manuscript/MoDE-PK_EJPS_completion_20261002'
def save(value):
 p=B/'finalization_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def command(args,cwd=R):
 print('RUN',args,flush=True);subprocess.run(args,cwd=cwd,check=True,env=os.environ|dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
try:
 last=None
 while True:
  d=json.loads((B/'budget_status.json').read_text());d.setdefault('completed',sum(j['status']=='completed' for j in d['jobs']));d.setdefault('failed',sum(j['status']=='failed' for j in d['jobs']));state=(d['completed'],d['failed'])
  if state!=last:print('budget completed/failed',state,flush=True);last=state
  save(dict(status='waiting_for_registered_runs',budget_completed=d['completed'],budget_total=360))
  if d['failed']:raise RuntimeError('A registered budget fit failed; investigate before export')
  extra=json.loads((B/'residual_augmentation_status.json').read_text()) if (B/'residual_augmentation_status.json').exists() else {'status':'running','jobs':[]}
  if any(j['status']=='failed' for j in extra['jobs']):raise RuntimeError('Centered augmentation fit failed')
  if d['completed']==360 and extra['status']=='completed':break
  time.sleep(20)
 for name in ['controls_status.json','simulation/status.json','simulation_crossed/status.json','residual_augmentation_status.json']:
  assert json.loads((B/name).read_text())['status']=='completed',name
 save(dict(status='aggregating'))
 for script in ['summarize_completion_suite.py','report_completion_suite.py','export_completion_manuscript.py']:
  command([sys.executable,'scripts/'+script])
 save(dict(status='compiling_pdfs'))
 for stem in ['main','supplementary']:
  command(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],M)
  command(['bibtex',stem],M)
  for _ in range(2):command(['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],M)
  log=(M/(stem+'.log')).read_text(errors='replace')
  assert 'undefined references' not in log.lower(),stem
  assert 'undefined citations' not in log.lower(),stem
 save(dict(status='packaging'))
 command([sys.executable,'scripts/package_completion_suite.py'])
 save(dict(status='completed_pending_visual_review',paper=str(M.relative_to(R)),report='results/summaries/completion_suite_20261002/REPORT_KO.md'))
except Exception as e:
 save(dict(status='failed',error=repr(e)));raise
