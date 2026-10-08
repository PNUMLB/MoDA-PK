from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[1]
def main():
 failed=[]
 for item in json.loads((R/'environment/required_inputs.json').read_text()):
  p=R/item['path'];status='missing' if not p.exists() else ('ok' if hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'] else 'hash mismatch')
  print(status,item['path'])
  if status!='ok':failed.append(item['path'])
 raise SystemExit(bool(failed))
if __name__=='__main__':main()
