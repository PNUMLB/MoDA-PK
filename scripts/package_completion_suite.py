"""Package verified aggregate evidence; never stage or push the Git repository."""
from pathlib import Path
import json,hashlib,shutil,zipfile,tarfile,subprocess
import pandas as pd
R=Path(__file__).resolve().parents[1];B=R/'results/runs/COMPLETION_SUITE/20261002';S=R/'results/summaries/completion_suite_20261002';M=R/'manuscript/MoDE-PK_EJPS_completion_20261002';D=R/'MoDE-PK_Paper_Materials_Completion_20261002'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((S/'completion_manifest.json').read_text())['status']=='completed'
assert (M/'main.pdf').exists() and (M/'supplementary.pdf').exists()
if not D.exists():shutil.copytree(R/'MoDE-PK_Paper_Materials_Drug_First_20261002',D)
for p in M.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(M)
 if p.suffix in ['.tex','.bib','.bbl','.cls','.sty','.bst']:
  q=D/'tex'/rel;q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.tex':
   text=p.read_text()
   if '\\begin{document}' in text and '\\graphicspath' not in text:text=text.replace('\\begin{document}','\\graphicspath{{../figures/}}\n\\begin{document}')
   q.write_text(text)
  else:shutil.copy2(p,q)
 elif p.suffix in ['.pdf','.png','.jpg']:
  dest='pdf' if p.name in ['main.pdf','supplementary.pdf','extended_analyses.pdf'] else 'figures';q=D/dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
A=D/'analysis/completion';A.mkdir(parents=True,exist_ok=True)
for p in S.iterdir():
 if p.is_file() and 'LOCAL_ONLY' not in p.name:
  shutil.copy2(p,A/p.name)
  if p.suffix in ['.csv','.xlsx']:shutil.copy2(p,D/'tables'/p.name)
  if p.suffix in ['.pdf','.png']:shutil.copy2(p,D/'figures'/p.name)
C=D/'code/completion'
for folder in ['src','scripts','tests']:
 for p in (R/folder).rglob('*.py'):
  if '__pycache__' in p.parts:continue
  q=C/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
E=D/'experiments/completion';E.mkdir(parents=True,exist_ok=True)
for p in [B/'registered_plan.json',B/'controls_plan.json',B/'residual_augmentation_plan.json',B/'residual_augmentation_status.json',B/'configuration_notes.json',B/'source_manifest.json',B/'augmentation_addendum_source_manifest.json',B/'environment.packages.json',B/'budget_status.json',B/'controls_status.json',B/'simulation_crossed/plan.json',B/'simulation_crossed/status.json']:
 name=('simulation_crossed_'+p.name) if p.parent.name=='simulation_crossed' else p.name;shutil.copy2(p,E/name)
shutil.copy2(R/'docs/COMPLETION_EXPERIMENTS_20261002.md',D/'guides/COMPLETION_EXPERIMENTS_20261002.md')
# This final snapshot is distinct from the source snapshot captured before execution.
with tarfile.open(B/'final_analysis_source.tar.gz','w:gz') as tar:
 for p in sorted(C.rglob('*.py')):tar.add(p,arcname=str(p.relative_to(C)))
manifest=dict(status='completed',model='MoDE-PK',primary_configuration_unchanged=True,new_fits=1020,reused_budget_fits=40,public_bundle_contains_query_data=False,initial_source_manifest_sha256=sha(B/'source_manifest.json'),final_analysis_source_sha256=sha(B/'final_analysis_source.tar.gz'),environment_sha256=sha(B/'environment.packages.json'),analysis_manifest_sha256=sha(S/'completion_manifest.json'),main_tex_sha256=sha(M/'main.tex'),main_pdf_sha256=sha(M/'main.pdf'),supplementary_tex_sha256=sha(M/'supplementary.tex'),supplementary_pdf_sha256=sha(M/'supplementary.pdf'),git_parent=subprocess.check_output(['git','-C',str(R/'MoDE-PK'),'rev-parse','HEAD'],text=True).strip())
(D/'manifests/completion_manifest.json').write_text(json.dumps(manifest,indent=2))
(D/'README_KO.md').write_text('''# MoDE-PK 논문·추가 실험 자료 — 2026-10-02

- pdf/main.pdf, pdf/supplementary.pdf: 추가 실험 반영 원고.
- analysis/completion/REPORT_KO.md: 실행 범위, 결과, 해석.
- tables/Completion_results.xlsx: cohort/약물/seed/실행별 수치.
- figures/: 본문·보충 및 신규 분석 그림.
- tex/: 원고 소스. 이 디렉토리에서 pdflatex를 실행한다.
- code/completion/: 실행·모델·집계·그림·원고 생성 코드의 최종 스냅샷.
- experiments/completion/: 등록 계획과 실행 상태, 환경 및 원본 source manifest.
- manifests/completion_manifest.json, FILE_INDEX.csv: 파일 계보와 checksum.

기존 원고 자료를 함께 보존하므로 analysis의 다른 하위 폴더와 예전 manifest는 해당 분석 시점의 기록이다. 현재 원고와 추가 실험은 completion 폴더를 기준으로 확인한다.

원자료·개별 query/참가자 예측·체크포인트는 포함하지 않는다. 서버 results/runs/COMPLETION_SUITE/20261002 및 기존 실행 경로에 보존했다. 재현에는 적법하게 확보한 데이터와 등록된 기존 split/실행 자산이 필요하다. 이 묶음은 원자료 없이 학습을 재실행하는 독립 데이터 배포본이 아니다.

기존 모델을 test 결과로 교체하지 않았다. 실제 새 외부 연구를 확보한 것으로 해석하지 않는다. Git 폴더는 별도의 ../MoDE-PK이며 add/commit/push는 수행하지 않았다.
''')
idx=[dict(path=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file() and p.name!='FILE_INDEX.csv'];pd.DataFrame(idx).to_csv(D/'FILE_INDEX.csv',index=False)
z=R/'deliverables/MoDE-PK_Paper_Completion_20261002.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for p in sorted(D.rglob('*')):
  if p.is_file():f.write(p,arcname=str(p.relative_to(D.parent)))
(R/'manuscript/CURRENT_MANUSCRIPT.md').write_text('''# Current manuscript — MoDE-PK

Current folder: `MoDE-PK_EJPS_completion_20261002`.

[Main](MoDE-PK_EJPS_completion_20261002/main.pdf) · [Supplementary](MoDE-PK_EJPS_completion_20261002/supplementary.pdf).

The primary drug-level comparisons remain unchanged. Additional budget, encoder, synthetic augmentation, mechanism, resampling, molecular-representation, and failure-stage analyses are complete. New architectures were not selected on test rankings. Previous releases are preserved.

Materials: ../MoDE-PK_Paper_Materials_Completion_20261002/. ZIP: ../deliverables/MoDE-PK_Paper_Completion_20261002.zip. Git changes not staged, committed, or pushed.
''')
print(z,z.stat().st_size)
