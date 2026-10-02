"""Fetch the pinned upstream repositories locally; never vendor them into Git."""
from pathlib import Path
import csv,subprocess
R=Path(__file__).resolve().parents[1]
def main():
 for row in csv.DictReader((R/'environment/upstream_repositories.csv').open()):
  if row['repository'] not in ['DeepCt','UniPKModel']:continue
  p=R/'third_party'/row['repository']
  if not p.exists():
   p.parent.mkdir(exist_ok=True);subprocess.run(['git','clone','--no-checkout',row['url'],str(p)],check=True)
  subprocess.run(['git','-C',str(p),'checkout','--detach',row['commit']],check=True)
  actual=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
  assert actual==row['commit'];print(row['repository'],actual)
if __name__=='__main__':main()
