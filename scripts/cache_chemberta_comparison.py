"""Pinned frozen ChemBERTa encoders; no PK labels used to generate embeddings."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,RobertaConfig,RobertaModel
from rdkit import Chem,rdBase
R=Path(__file__).resolve().parents[1];O=R/'data/features/molecular_comparison_20261002'
PINS={'chemberta_mlm':('DeepChem/ChemBERTa-77M-MLM','ed8a5374f2024ec8da53760af91a33fb8f6a15ff'),'chemberta_mtr':('DeepChem/ChemBERTa-77M-MTR','66b895cab8adebea0cb59a8effa66b2020f204ca')}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);O.mkdir(parents=True,exist_ok=True)
 base=R/'data/processed/full16_v1/embeddings/MANIFEST.json';manifest=json.loads(base.read_text());ref=R/'data/metadata/provenance_groups_20260925/source_traceable/dataset_reference.json';drugs=sorted(json.loads(ref.read_text())['drug_ids']);smiles=[manifest['smiles'][d] for d in drugs]
 raw=pd.read_csv(R/'data/processed/full16_v1/primary_full16.csv',usecols=['dataset','smiles_for_model']).drop_duplicates()
 for d,s in zip(drugs,smiles):
  original=raw[raw.dataset==d].smiles_for_model.unique().tolist();assert len(original)==1 and Chem.MolToSmiles(Chem.MolFromSmiles(original[0]),isomericSmiles=True)==Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True),d
 for name,(repo,revision) in PINS.items():
  dest=O/(name+'.npz')
  if dest.exists():assert (O/(name+'.json')).exists();continue
  path=Path(snapshot_download(repo_id=repo,revision=revision,allow_patterns=['config.json','pytorch_model.bin','*token*.json','vocab.json','merges.txt'],max_workers=4))
  tok=AutoTokenizer.from_pretrained(path,local_files_only=True);config=RobertaConfig.from_pretrained(path,local_files_only=True)
  state=torch.load(path/'pytorch_model.bin',map_location='cpu',weights_only=True);encoder={k[len('roberta.'):]:v for k,v in state.items() if k.startswith('roberta.')}
  m=RobertaModel(config,add_pooling_layer=False);expected=set(m.state_dict());ignored={k:encoder.pop(k) for k in list(encoder) if k not in expected and k in ['embeddings.position_ids','embeddings.token_type_ids']};m.load_state_dict(encoder,strict=True);m.eval();m.requires_grad_(False)
  def embed(ss):
   batch=tok(ss,padding=True,truncation=False,return_tensors='pt',return_special_tokens_mask=True);special=batch.pop('special_tokens_mask');mask=batch['attention_mask'].bool()&~special.bool();assert mask.sum(1).min()>0
   assert batch['input_ids'].shape[1] <= config.max_position_embeddings-config.pad_token_id-1
   with torch.no_grad():h=m(**batch).last_hidden_state;z=(h*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]
   return z.numpy(),batch,mask
  z,batch,mask=embed(smiles);single=np.concatenate([embed([s])[0] for s in smiles]);assert np.isfinite(z).all();assert np.allclose(z,single,atol=2e-6,rtol=2e-6)
  np.savez_compressed(dest,drugs=np.array(drugs),embeddings=z,smiles=np.array(smiles))
  audit=[]
  for i,d in enumerate(drugs):audit.append(dict(drug_id=d,smiles=smiles[i],tokens=int(batch['attention_mask'][i].sum()),pooled_tokens=int(mask[i].sum()),unknown_tokens=int(((batch['input_ids'][i]==tok.unk_token_id)&mask[i]).sum())))
  old=R/'data/processed/full16_v1/embeddings/chemberta.npz';previous=None
  if name=='chemberta_mtr':
   with np.load(old) as x:v=dict(zip(x['drugs'].astype(str),x['embeddings']))
   delta=np.stack([v[d][:z.shape[1]] for d in drugs])-z;previous=dict(path=str(old.relative_to(R)),sha256=sha(old),max_difference=float(abs(delta).max()),note='Recomputed pinned encoder; old cache not substituted')
  record=dict(repo=repo,revision=revision,model_files={p.name:sha(p) for p in path.iterdir() if p.is_file()},dimension=z.shape[1],pooling='mean final hidden states over attention-valid non-special tokens',frozen=True,uses_pk_targets=False,smiles_rule='Exact embedding-manifest SMILES; RDKit isomeric canonical equivalence checked against raw SMILES; no new salt or stereo processing',rdkit_version=rdBase.rdkitVersion,raw_smiles_graph_equivalence=True,source_manifest_sha256=sha(base),dataset_reference_sha256=sha(ref),encoder_keys=len(encoder),ignored_deterministic_buffers=list(ignored),unused_head_keys=[k for k in state if not k.startswith('roberta.')],trainable_encoder_parameters=0,encoder_parameters=sum(p.numel() for p in m.parameters()),embedding_sha256=sha(dest),batch_single_max_difference=float(abs(z-single).max()),audit=audit,previous_cache_comparison=previous)
  (O/(name+'.json')).write_text(json.dumps(record,indent=2));print(name,'shape',z.shape,'unknown tokens',sum(x['unknown_tokens'] for x in audit),'old',previous,flush=True)
if __name__=='__main__':main()
