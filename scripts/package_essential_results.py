"""Package aggregate manuscript evidence, excluding participant/query records."""
from pathlib import Path
import shutil,json,hashlib,zipfile,subprocess,tarfile
import pandas as pd
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/essential_20261002';B=R/'results/runs/ESSENTIAL_20261002';M=R/'manuscript/MoDE-PK_EJPS_essential_20261002';D=R/'MoDE-PK_Paper_Materials_Essential_20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not D.exists():shutil.copytree(R/'MoDE-PK_Paper_Materials_20261002',D)
for p in M.glob('*.tex'):
 t=p.read_text()
 if p.name in ['main.tex','supplementary.tex','extended_analyses.tex'] and '\\graphicspath' not in t:t=t.replace('\\begin{document}','\\graphicspath{{../figures/}}\n\\begin{document}')
 (D/'tex'/p.name).write_text(t)
for n in ['main.pdf','supplementary.pdf','extended_analyses.pdf','Ten_seed_curve_atlas.pdf']:shutil.copy2(M/n,D/'pdf'/n)
A=D/'analysis/essential';A.mkdir(exist_ok=True)
for p in S.iterdir():
 if p.is_file() and 'LOCAL_ONLY' not in p.name:
  shutil.copy2(p,A/p.name)
  if p.suffix in ['.csv','.xlsx']:shutil.copy2(p,D/'tables'/p.name)
for p in S.glob('E01_validation_selection.*'):shutil.copy2(p,D/'figures'/p.name)
scripts=['run_essential_selection.py','audit_essential_results.py','probe_essential_floor.py','summarize_essential.py','export_essential_manuscript.py','package_essential_results.py']
for n in scripts:
 shutil.copy2(R/'scripts'/n,D/'code'/n)
 shutil.copy2(R/'scripts'/n,R/'MoDE-PK/scripts'/n)
E=D/'experiments/essential';E.mkdir(exist_ok=True)
shutil.copy2(B/'plan.json',E/'plan.json')
for seed in range(1,6):
 for name in ['manifest.json','selection.json','replay_verification.json']:
  shutil.copy2(B/f'E01__s{seed}'/name,E/f'seed{seed}_{name}')
shutil.copy2(R/'Essential_Experiments.md',D/'guides/Essential_Experiments_requested.md')
(D/'README_KO.md').write_text('''# MoDE-PK 논문 자료 — Essential 보완, 2026-10-02

현재 원고는 pdf/main.pdf 및 pdf/supplementary.pdf다. tex/는 그림을 ../figures/에서 참조한다. 기존 주요 성능표를 유지하고 Essential_Experiments.md의 E02 무결성 감사와 E01 S5 선택 민감도를 추가했다.

- analysis/essential/REPORT_KO.md: 결과 해석과 확인 범위.
- tables/Essential_results.xlsx: 2×2 결과, seed별 지표, 선택 epoch, 약물별 지표, 감사 결과.
- figures/E01_validation_selection.pdf: 다섯 seed의 validation 선택 곡선.
- experiments/essential/: 실제 실행 manifest와 재현 확인.
- code/: 관련 재현·집계·원고 생성 코드. 전체 모델 코드는 별도 MoDE-PK Git 폴더.
- manifests/essential_manifest.json: 주요 파일 checksum과 테스트 기록.

개별 원자료·query 예측·checkpoint는 포함하지 않는다. 서버의 results/runs/ESSENTIAL_20261002에 보존했다. 재현에는 적법하게 확보한 원자료와 기존 실행 자산이 필요하다. 기존 묶음에서 이어온 실험 스냅샷은 해당 시점 기록이다. 별도 RESAMPLING 실행은 현재 완료됐지만 그 결과로 이 원고의 주 표를 대체하지 않았다. 저자·지원·윤리 선언은 저자의 확인 사항을 보존했다.
''')
# Reproducibility snapshot is internal; it includes code only, not source data.
with tarfile.open(B/'code_snapshot.tar.gz','w:gz') as tar:
 for folder in ['src','scripts']:
  for p in sorted((R/folder).rglob('*.py')):
   if '__pycache__' not in p.parts:tar.add(p,arcname=str(p.relative_to(R)))
packages=subprocess.check_output(['python','-m','pip','list','--format=json'],stderr=subprocess.DEVNULL);(B/'environment.packages.json').write_bytes(packages)
manifest=dict(scope='Essential E02 + E01; no change to primary selection or tables',model='MoDE-PK',tests='11 passed: metric, quality_metrics, drug_metrics, traceable_cohort, deepct_concentration_units',git_release_parent=subprocess.check_output(['git','-C',str(R/'MoDE-PK'),'rev-parse','HEAD'],text=True).strip(),internal_code_snapshot_sha256=sha(B/'code_snapshot.tar.gz'),environment_sha256=sha(B/'environment.packages.json'),files={str(p.relative_to(R)):sha(p) for p in [R/'Essential_Experiments.md',M/'main.tex',M/'supplementary.tex',M/'essential_results.tex',M/'main.pdf',M/'supplementary.pdf',S/'E01_2x2_summary.csv',S/'E02_status.json']})
(D/'manifests/essential_manifest.json').write_text(json.dumps(manifest,indent=2));(B/'completion_manifest.json').write_text(json.dumps(manifest,indent=2))
idx=[dict(path=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='FILE_INDEX.csv'];pd.DataFrame(idx).to_csv(D/'FILE_INDEX.csv',index=False)
z=R/'deliverables/MoDE-PK_Paper_Essential_20261002.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(D.rglob('*')):
  if p.is_file():f.write(p,arcname=str(p.relative_to(D.parent)))
(R/'manuscript/CURRENT_MANUSCRIPT.md').write_text('''# Current manuscript — MoDE-PK

Current folder: `MoDE-PK_EJPS_essential_20261002`.

[Main PDF](MoDE-PK_EJPS_essential_20261002/main.pdf) · [Supplementary](MoDE-PK_EJPS_essential_20261002/supplementary.pdf).

E02 integrity audit and five-seed E01 validation-selection sensitivity are complete. Primary ten-seed tables and selection rule are unchanged. Previous release is preserved at MoDE-PK_EJPS_release_20261002.

Paper materials: ../MoDE-PK_Paper_Materials_Essential_20261002/. Git code: ../MoDE-PK/ (not staged, committed, or pushed by this task). Separate resampling runs completed; their results are not substituted into this manuscript.
''')
print(z,z.stat().st_size)
