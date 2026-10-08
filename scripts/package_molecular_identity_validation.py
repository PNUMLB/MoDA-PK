"""Package reviewed manuscript and this input-validation result without raw source data."""
from pathlib import Path
import json,hashlib,shutil,zipfile
R=Path(__file__).resolve().parents[1];D=R/'manuscript/MoDE-PK_EJPS_identity_validation_20261002';S=R/'results/summaries/molecular_identity_20261002';B=R/'results/runs/MOLECULAR_IDENTITY/20261002';P=R/'MoDE-PK_Identity_Validation_20261002';Z=R/'deliverables/MoDE-PK_Identity_Validation_20261002.zip'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P.mkdir(exist_ok=False)
for file in D.rglob('*'):
 if file.is_file() and file.suffix.lower() in ['.tex','.pdf','.bib','.cls','.sty','.csv','.json','.md','.py']:
  dest=P/'manuscript'/file.relative_to(D);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
for file in S.iterdir():
 if file.is_file() and file.suffix in ['.csv','.xlsx','.md','.json','.pdf','.png']:
  dest=P/'analysis'/file.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
for file in [B/'registered_plan.json',B/'plan.json',B/'status.json',B/'code_manifest.json',B/'source_snapshot.tar.gz',B/'environment.txt']:
 dest=P/'provenance'/file.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
files=list((R/'scripts').glob('*molecular_identity*.py'))+[R/'src/mhap_pk/models/priors/molecular_identity.py',R/'tests/test_molecular_identity.py',R/'docs/MOLECULAR_IDENTITY_VALIDATION_20261002.md']
for f in files:
 d=P/'code'/f.relative_to(R);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,d)
(P/'README_KO.md').write_text('''# MoDE-PK 분자·Drug-ID 결합 검증

- manuscript/main.pdf, supplementary.pdf: 최신 원고. Known/Joint 동일6profile 그림을 각각 본문에 배치.
- analysis/molecular_identity_results.xlsx, REPORT_KO.md: 단독입력2조건+결합조건,10seeds,약물별지표·paired효과·관측효과·편향분해.
- provenance: 신규100fits 등록·환경·훈련코드 snapshot. 기존200fits 출처는 run_inventory.csv.
- code: 이번 추가모델·실행·집계·그림·원고내보내기·검증 코드.

기존 주모델은 유지한다. 결합 모델의 개선은 일부 조건에서 관측되며 Joint에서 Drug-ID 단독을 넘지 못했다. 새 외부 검증이나 신규약물 전반 우월성의 근거는 아니다.

로컬 검토용 묶음이다. 익명화된 study-scoped profile 식별자가 일부 진단에 포함되어 있어 공개 Git에 그대로 올리는 자료가 아니다. 원자료, checkpoint, query-level prediction parquet는 포함하지 않으며 서버 경로와 checksum을 참조한다. 기존 데이터 및 fitting artifacts가 있어야 학습을 재실행할 수 있다. code/provenance만으로 독립 실행 가능한 데이터 배포본이라고 주장하지 않는다.

원래 그림·표 자산을 보존하므로 PDF에 인용되지 않는 이전 그림도 폴더에 남아 있다. 실제 제출 구성은 main.tex와 supplementary.tex가 결정한다.
''')
manifest={'status':'completed','experiment':str(B.relative_to(R)),'summary':str(S.relative_to(R)),'manuscript':str(D.relative_to(R)),'canonical_prediction':{'path':str((S/'canonical_predictions.parquet').relative_to(R)),'sha256':sha(S/'canonical_predictions.parquet')},'files':{str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()}}
(P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
Z.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(Z,'w',zipfile.ZIP_DEFLATED) as z:
 for p in P.rglob('*'):
  if p.is_file():z.write(p,str(p.relative_to(P.parent)))
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None
 for path,h in manifest['files'].items():assert hashlib.sha256(z.read(P.name+'/'+path)).hexdigest()==h
print(json.dumps({'folder':str(P),'zip':str(Z),'MiB':Z.stat().st_size/2**20,'verified_files':len(manifest['files'])},indent=2))
