"""Compile both documents without importing supplementary bibliography citations."""
from pathlib import Path
import sys,subprocess,re
D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'manuscript/MoDE-PK_EJPS_hybrid_primary_20261002'
for stem in ['supplementary','main']:
 if stem=='main':
  labels=[s for s in (D/'supplementary.aux').read_text().splitlines() if s.startswith('\\newlabel')];(D/'supplementary_labels.aux').write_text('\\relax\n'+'\n'.join(labels)+'\n')
 commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex'],['bibtex',stem]]+[['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex']]*3
 for c in commands:
  r=subprocess.run(c,cwd=D,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  with (D/(stem+'.build.log')).open('a') as f:f.write(r.stdout)
  if r.returncode:print(r.stdout[-5000:]);raise SystemExit(r.returncode)
 log=(D/(stem+'.log')).read_text();assert not re.search(r'(undefined references|undefined citations|Citation .* undefined|Reference .* undefined|Overfull|multiply defined|Label\(s\) may have changed)',log),stem
 print(stem,'built; references and layout checks passed',flush=True)
