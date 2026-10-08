"""Verify, publish local manuscript pointer, and build a private review bundle."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,re
import pandas as pd
import pymupdf
R=Path(__file__).resolve().parents[1];S=R/'results/summaries/hybrid_primary_20261002';D=R/'manuscript/MoDE-PK_EJPS_hybrid_primary_20261002';G=R/'MoDE-PK';B=R/'MoDE-PK_Hybrid_Primary_20261002';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 for p in [S/'analysis_manifest.json',S/'additional_manifest.json',S/'baseline_reuse_manifest.json',S/'representation_manifest.json',S/'cross_model_representation_manifest.json']:
  assert json.loads(p.read_text())['status']=='completed'
 checks={}
 for stem in ['main','supplementary']:
  log=(D/(stem+'.log')).read_text();assert not re.search(r'(undefined references|undefined citations|Citation .* undefined|Reference .* undefined|Overfull|multiply defined)',log),stem
  doc=pymupdf.open(D/(stem+'.pdf'));text='\n'.join(p.get_text() for p in doc);assert 'MoDE-PK' in text and '2.1253' in text;checks[stem]=dict(pages=len(doc),sha256=sha(D/(stem+'.pdf')),references_resolved=True,no_overfull_boxes=True)
  (D/'visual_checks').mkdir(exist_ok=True)
  for i,p in enumerate(doc):
   if (stem=='main' and any(x in p.get_text() for x in ['Table 1','Table 2','Table 3','Figure 1','Figure 2','Figure 3'])) or (stem=='supplementary' and 'UMAP' in p.get_text() and p.get_images()):p.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(D/'visual_checks'/f'{stem}_{i+1}.png')
 (D/'build_verification.json').write_text(json.dumps(checks,indent=2))
 # Current public code only; no staging, committing or pushing.
 names=[p.name for p in (R/'scripts').glob('*hybrid*.py')]
 names.append('build_hybrid_manuscript.py') if 'build_hybrid_manuscript.py' not in names else None
 for name in names:shutil.copy2(R/'scripts'/name,G/'scripts'/name);shutil.copy2(R/'scripts'/name,D/'revision_code'/name)
 for name in ['summarize_training_scope_extension.py','run_molecular_identity_validation.py','summarize_molecular_identity_validation.py']:
  if not (G/'scripts'/name).exists():shutil.copy2(R/'scripts'/name,G/'scripts'/name)
 B.mkdir(exist_ok=True)
 for p in list(B.iterdir()):
  # Build only in this declared release directory; never remove prior research runs.
  if p.is_file() and p.name=='bundle_manifest.json':p.unlink()
 for name in ['main.pdf','supplementary.pdf','main.tex','supplementary.tex','main.bbl','supplementary.bbl','supplementary_labels.aux','references.bib','elsarticle.cls','elsarticle-harv.bst','analysis_lineage.json','build_verification.json']:
  if (D/name).exists():shutil.copy2(D/name,B/name)
 shutil.copytree(D/'tables',B/'tables',dirs_exist_ok=True)
 (B/'figures').mkdir(exist_ok=True)
 for p in S.glob('*.pdf'):shutil.copy2(p,B/'figures'/p.name);shutil.copy2(p,B/p.name)
 shutil.copytree(S,B/'analysis',dirs_exist_ok=True)
 # This review bundle includes derived predictions and deidentified support/dose plot inputs.
 # Raw source datasets and trained checkpoints remain at their immutable server locations.
 code=B/'code';shutil.copytree(R/'src',code/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(code/'scripts').mkdir(exist_ok=True,parents=True)
 for p in (R/'scripts').glob('*.py'):shutil.copy2(p,code/'scripts'/p.name)
 shutil.copy2(G/'scripts/train.py',code/'scripts/train.py');shutil.copy2(G/'README.md',code/'README.md')
 shutil.copytree(G/'configs',code/'configs',dirs_exist_ok=True);shutil.copytree(G/'environment',code/'environment',dirs_exist_ok=True,ignore=shutil.ignore_patterns('*copy_manifest*','*release_manifest*'))
 for name in ['requirements.txt','pyproject.toml','THIRD_PARTY_NOTICES.md']:
  if (G/name).exists():shutil.copy2(G/name,code/name)
 shutil.copytree(R/'tests',code/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
 for name in ['MODEL_SPEC.md','MODEL_ARCHITECTURE_KO.md']:shutil.copy2(G/'docs'/name,B/name)
 shutil.copy2(R/'docs/MODE_PK_HYBRID_ADOPTION.md',B/'MODEL_CHANGE.md')
 manifests=B/'experiment_manifests';manifests.mkdir(exist_ok=True)
 for family in ['HYBRID_PRIMARY','MOLECULAR_IDENTITY']:
  root=R/'results/runs'/family/'20261002';dest=manifests/family;dest.mkdir(exist_ok=True)
  for name in ['registered_plan.json','status.json','code_manifest.json','input_manifest.json','plan.json','source_snapshot.tar.gz','environment.txt']:
   if (root/name).exists():shutil.copy2(root/name,dest/name)
 # Preserve per-fit configurations and selected-epoch records without checkpoints/log bulk.
 inv=pd.read_csv(S/'new_run_inventory.csv')
 for rid in inv.run_id:
  root=R/rid;dest=manifests/'fits'/Path(rid).relative_to('results/runs/HYBRID_PRIMARY/20261002');dest.mkdir(parents=True,exist_ok=True)
  for name in ['config.resolved.json','run_manifest.json','selection.json','verification.json','population.json','artifacts.json']:
   if (root/name).exists():shutil.copy2(root/name,dest/name)
 for root in (R/'results/runs/MOLECULAR_IDENTITY/20261002').glob('*__s*__molecular'):
  dest=manifests/'primary_fits'/root.name;dest.mkdir(parents=True,exist_ok=True)
  for name in ['config.resolved.json','run_manifest.json','selection.json','verification.json','population.json','artifacts.json']:
   if (root/name).exists():shutil.copy2(root/name,dest/name)
 summary=pd.read_csv(S/'primary_cohort_summary.csv');summary=summary[(summary.protocol=='first_k')&(summary.k==2)];lines=['# MoDE-PK 결합 모델 최종 분석','', '**현재 Ours: Uni-Mol + Drug-ID concatenation, parameter/state update ON, relation OFF.**','', '주 모델 100 fits의 예측을 검증해 재사용했고, 보완 분석 2,230 fits를 새로 완료했다. 전 실험 seed 1–10. 기존 비교모델 예측의 동일 query와 원본 계보를 검증했다.','', '| 평가 | 모델 | SB-hRMSE (평균 ± SD) |','|---|---|---:|']
 labels={'ours':'MoDE-PK (Ours)','poppk_vi_map':'PopPK + MAP','dcm':'DCM','unipk':'Uni-PK','deepct':'DeepCt'}
 for a in summary.itertuples():lines.append(f'| {a.track} | {labels[a.condition]} | {a.sb_hrmse_mean:.4f} ± {a.sb_hrmse_std:.4f} |')
 lines+=['','Known은 기존 0.4609보다 약간 높고, Joint는 기존 2.2993보다 낮다. Compact Drug-ID-only Joint 1.8908보다 우수한 것은 아니다. 내부 결과 확인 후 채택한 모델이며 독립 외부 검증으로 주장하지 않는다.','', '## 파일','', '- main.pdf / supplementary.pdf: 현재 결합 모델 기준 원고.','- analysis/MoDE_PK_hybrid_results.xlsx 및 tables/: query 예측에서 재집계한 약물별·seed별·ablation·pooling·CV·budget·진단 결과.','- figures/: 새 Ours의 구조, Known/Joint 동일 사례 곡선, k 분석, 약물별 horizon 및 query atlas, input-control/cross-model UMAP.','- analysis/*predictions.parquet: 분석 재생성용 예측.','- analysis/plot_support_doses_LOCAL_ONLY.json: 그림에 필요한 관측·투약 입력.','- code/: 모델과 분석/그림 생성 코드. experiment_manifests/: 실제 설정과 checksum, 학습 partition 및 선택 epoch.','', '## 재생성','', '서버 원자료 없이 저장된 예측·투약/관측 입력으로 주 curve와 history/horizon figure 재생성:','', '```bash','python code/scripts/plot_hybrid_primary.py --summary analysis','```','', '원고를 빌드하려면 PDF·tex·bib가 있는 이 폴더에서 supplementary → main 순서로 pdflatex/bibtex를 실행한다. 내부 supplementary 참조는 xr로 연결된다. 학습 재실행과 checkpoint 표현 재추출은 별도 서버 데이터·checkpoint 및 comparator sources가 필요하다.','', '이 검토용 bundle에는 파생 예측과 익명 source-scoped record 식별자가 포함되어 있으므로 공개 Git용 파일이 아니다. 출처별 재배포 조건을 확인한 후 공유한다. Git용 MoDE-PK/에는 결과·원고·환자 단위 파일을 넣지 않았다.','', '이전 molecular-only/frozen-prior/합성 탐색 결과는 별도 보존되며 새 Ours로 이름을 바꾸지 않았다. 현재 본문·보충자료에 남긴 모든 Ours 분석은 결합 모델과 명시된 새 ablation에 대응한다.']
 (B/'README_KO.md').write_text('\n'.join(lines)+'\n');(S/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
 manifest={str(p.relative_to(B)):sha(p) for p in B.rglob('*') if p.is_file()};(B/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2));out=R/'deliverables'/f'{B.name}.zip';out.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(B.rglob('*')):
   if p.is_file():z.write(p,p.relative_to(B.parent))
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 (R/'manuscript/CURRENT_MANUSCRIPT.md').write_text(f'# Current manuscript — MoDE-PK molecular + identity\n\n[Main PDF]({D.name}/main.pdf) · [Supplementary PDF]({D.name}/supplementary.pdf) · [Results workbook](../results/summaries/hybrid_primary_20261002/MoDE_PK_hybrid_results.xlsx) · [Review ZIP](../deliverables/{out.name}).\n\nCurrent Ours is the concatenation model, both updates, relations OFF. 100 primary fits verified/reused; 2,230 supporting fits completed. All retained Ours analyses use this model or named retrained controls. Earlier model analyses remain separately archived. No Git commit or push performed.\n')
 (R/'docs/CURRENT_HYBRID_STATUS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(checks,indent=2));print(out,'MiB',out.stat().st_size/2**20,'files',len(manifest)+1)
if __name__=='__main__':main()
