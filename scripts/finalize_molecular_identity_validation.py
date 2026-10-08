"""Record interpretation, data checks and paired-query provenance before export."""
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from mhap_pk.data.traceable import load_profiles
from mhap_pk.data import KS
S=R/'results/summaries/molecular_identity_20261002';D=R/'manuscript/MoDE-PK_EJPS_identity_validation_20261002'
ps,meta,_=load_profiles();inv=pd.read_csv(S/'run_inventory.csv');checks=[]
for _,row in inv.iterrows():
 a=json.loads((R/row.run_id/'artifacts.json').read_text());groups=[set(map(tuple,meta.iloc[a[k]][['source_id','participant_id']].to_numpy())) for k in ['train','validation','test']]
 assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
 fitted=set(meta.iloc[a['train']].drug_id);assert set(a['scaler']['fitted_drugs'])==fitted
 if row.fold!='known':assert fitted.isdisjoint(set(meta.iloc[a['test']].drug_id))
 for idx in a['test']:
  p=ps[idx];assert p['query'].time.min()>p['episodes'][list(KS).index(4)].support.time.max()
 checks.append(dict(run_id=row.run_id,recorded_participant_roles_disjoint=True,scaler_drugs_training_only=True,query_after_fourth_time=True,unknown_drugs_absent_from_training=row.fold!='known'))
pd.DataFrame(checks).to_csv(S/'partition_time_checks.csv',index=False)
cs=pd.read_csv(S/'cohort_summary.csv');ds=pd.read_csv(S/'drug_summary.csv');bias=pd.read_csv(S/'profile_bias_shape.csv');paired=pd.read_csv(S/'paired_drug_summary.csv');inv=pd.read_csv(S/'run_inventory.csv')
rows=[]
for (track,c,d),g in bias.groupby(['track','condition','drug_id']):
 # Query-weighted profiles within each participant, then participant and seed balance.
 g=g.copy()
 for col in ['mse','bias_squared','centered_mse']:g[col+'_sum']=g[col]*g.queries
 t=g.groupby(['seed','source_id','participant_id'])[['mse_sum','bias_squared_sum','centered_mse_sum','queries']].sum()
 for col in ['mse','bias_squared','centered_mse']:t[col]=t[col+'_sum']/t.queries
 v=t.groupby('seed')[['mse','bias_squared','centered_mse']].mean().mean()
 rows.append(dict(track=track,condition=c,drug_id=d,**v.to_dict(),bias_squared_fraction=v.bias_squared/v.mse))
bd=pd.DataFrame(rows);bd.to_csv(S/'bias_shape_drug_summary.csv',index=False)
selection=inv.assign(at_initialization=inv.epoch==0,at_ceiling=inv.epoch==120).groupby(['condition','fold']).agg(fits=('seed','size'),initialization_selected=('at_initialization','sum'),ceiling_selected=('at_ceiling','sum'),selected_epoch_mean=('epoch','mean')).reset_index();selection.to_csv(S/'selection_summary.csv',index=False)
text='''

## 최종 판단

- 결합 입력은 Uni-Mol 단독 대비 Known SB-hRMSE를 0.4876→0.4664, Joint를 2.3486→2.1253으로 낮췄다. Drug-ID 단독 대비 Known은 낮지만 Joint는 1.8908→2.1253으로 높다. 따라서 두 단독입력을 모두 넘어선 보편적 상호보완성은 확인되지 않았다.
- Known은 Uni-Mol 대비 5/11 약물만 개선했으므로 계층평균 개선을 전약물 개선으로 해석하지 않는다. Joint는 9/11 개선, 1개 악화, 1개 동일이다. Drug-ID 대비 Joint 개선은 2/11이다.
- 같은 결합 checkpoint에서 k0→k2는 Known 0.4811→0.4664, Joint 2.1291→2.1253이다. 관측효과는 Known에서 더 뚜렷하고 Joint 평균 변화는 작다. 이는 고정 checkpoint 진단이며 재학습 ablation이 아니다.
- Warfarin Joint는 결합 4.3653, Uni-Mol 4.3374, Drug-ID 4.6205이고, Xanthohumol은 각각 3.8364, 4.2612, 2.6759이다. 두 약물의 실패를 해결했다고 볼 수 없다. 원래 full MoDE-PK와는 relation loss 조건이 다르다.
- Unknown Drug-ID의 좋은 결과는 새로운 약물의 구분자를 학습한 효과가 아니다. Unknown에는 학습약물 ID 평균을 제공한다. 화학적 외삽보다 이 fallback이 유리한 이유는 아직 인과적으로 분리되지 않았다.
- Joint remifentanil에서는 세 조건 모두 validation이 epoch0을 선택한다. 같은 점수는 학습된 일반화가 확인되었다는 뜻이 아니다. 모든 fold의 초기화/상한 선택 수를 별도로 제공한다.
- 결합 모델 용량은 44,045–46,093개로 Uni-Mol 24,077개, Drug-ID 27,661–29,709개보다 크다. 같은 epoch·학습률·validation 규칙을 맞췄지만 용량까지 같은 비교는 아니다.
- 기존 MoDE-PK를 유지한다. 신규 결합 모델은 입력 ablation에 추가하고, test 평균이 일부 좋다는 이유로 최종 구조를 바꾸지 않는다. 넓은 unknown 일반화 우월성은 주장하지 않는다.

검증: 신규100개 checkpoint 재로드 예측 일치, 기존200개 원본 예측/체크포인트 hash 확인, 300개 fitting artifact 일치 및 기록상 participant split 중복 차단, 1,200개 조건의 동일 query 확인. 검증은 기록된 linkage에 한하며 알려지지 않은 동일인 연결까지 입증하지 않는다. 새 외부 평가 자료는 추가하지 않았다.
'''
report=S/'REPORT_KO.md';report.write_text(report.read_text()+text)
# Preserve the balanced drug diagnostic and selection audit in the review workbook.
with pd.ExcelWriter(S/'molecular_identity_results.xlsx',mode='a',engine='openpyxl',if_sheet_exists='replace') as w:
 bd.to_excel(w,sheet_name='Bias shape drug summary',index=False);selection.to_excel(w,sheet_name='Selection audit',index=False);pd.DataFrame(checks).to_excel(w,sheet_name='Partition time checks',index=False)
main=(D/'main.tex').read_text()
paragraph='''In the relation-OFF input comparison, concatenating Uni-Mol and drug identity reduced SB-hRMSE from 0.4876 to 0.4664 in Known-drug forecasting and from 2.3486 to 2.1253 under joint holdout, relative to Uni-Mol alone. Drug identity alone remained better under joint holdout (1.8908). The combined input improved mean drug log-RMSE for five of eleven known drugs and nine of eleven held-out drugs relative to Uni-Mol alone; relative to drug identity alone, it improved only two held-out drugs. Thus the aggregate improvement does not establish uniform drug-level gains or superiority over both single-input controls. The original full model is retained, and the combined input is reported as a larger-capacity exploratory ablation.\n\n'''
anchor='The input ablation in Table~'
assert anchor in main;main=main.replace(anchor,paragraph+anchor,1)
main=main.replace('Drug identity (UNK for held-out drugs)', 'Drug identity (training-ID mean for held-out drugs)')
(D/'main.tex').write_text(main)
supp=(D/'supplementary.tex').read_text()
extra='''For the combined-input checkpoint, removing concentration observations yields SB-hRMSE 0.4811 in Known-drug forecasting and 2.1291 in joint holdout, compared with 0.4664 and 2.1253 at $k=2$. These are same-checkpoint history diagnostics, not retrained component ablations. Warfarin remains inaccurate (drug log-RMSE 4.3653), while Xanthohumol improves relative to Uni-Mol alone but remains worse than drug identity alone (3.8364 versus 2.6759). For both drugs, more than 98\\% of the combined model's participant-balanced log-MSE is the within-profile squared-bias component; the residual centered error is a descriptive diagnostic rather than evidence of accurate temporal forecasting. In the remifentanil fold all three input conditions select epoch 0. Trainable parameter counts are 24,077 for Uni-Mol, 27,661--29,709 for drug identity, and 44,045--46,093 for concatenation, depending on the number of training drugs.\n\n'''
# Verify numeric claim rather than silently insert a hard-coded interpretation.
z=bd[(bd.track=='Joint drug–source')&(bd.condition=='molecular_identity')&bd.drug_id.isin(['warfarin','xanthohumol'])];assert (z.bias_squared_fraction>.98).all()
figure=r'''
\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{Input_paired_drug_effects.pdf}
\caption{Paired drug-level effects of concatenating Uni-Mol and drug identity, with both relation losses OFF. Points are mean differences over ten paired training seeds and bars are one seed SD, not confidence intervals. Negative values favor concatenation. All eleven drugs are shown separately in both evaluation settings; the mean across drugs is not used to select displayed cases.}\label{fig:identity-paired}
\end{figure}
'''
import shutil
for name in ['Input_paired_drug_effects.pdf','Input_drug_metrics.pdf']:shutil.copy2(S/name,D/name)
supp=supp.replace('The accompanying workbook reports drug-level MAE',extra+figure+'\nThe accompanying workbook reports drug-level MAE');(D/'supplementary.tex').write_text(supp)
print('300 partition/time checks passed; interpretation exported.')
