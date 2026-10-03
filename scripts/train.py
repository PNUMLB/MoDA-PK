"""One new fit with the fixed dataset/split and manuscript concentration convention."""
from pathlib import Path
import argparse,json,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',default='known');p.add_argument('--seed',type=int,default=1);p.add_argument('--model',choices=['molecular_identity','molecular_relation','molecular','no_molecule','drug_id','ecfp','dcm','deepct','unipk'],default='molecular_identity');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data import representations,make_batch
 from mhap_pk.training.population import fit_population
 from mhap_pk.training.traceable_panel import run
 from complete_comparison_refresh import build
 from mhap_pk.models.priors.molecular_identity import build_joint_input
 torch.set_num_threads(1);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((R/'configs/experiments/source_traceable_20260925.json').read_text());cfg['additional_inference_k']=[3];cfg['model_id']=a.model;cfg['model_description']='MoDE-PK molecular+identity, both updates, relationOFF' if a.model=='molecular_identity' else 'Comparator or earlier condition: '+a.model;(out/'plan.json').write_text(json.dumps(cfg,indent=2))
 ps,meta,_=load_profiles();tr,_,_=indices_for_split(meta,a.fold);z,ids,_=representations(meta,tr,'unimol',17);_,cal=fit_population(make_batch(ps,meta,tr,z,ids));folder=out/'calibration'/a.fold;folder.mkdir(parents=True);(folder/'population.json').write_text(json.dumps(cal,indent=2,default=str))
 mode='molecular' if a.model=='molecular_identity' else a.model
 builder=build_joint_input if a.model=='molecular_identity' else build
 run(a.fold,a.seed,mode,output_root=out,model_builder=builder)
 manifest_path=out/f'{a.fold}__s{a.seed}__{mode}'/'run_manifest.json'
 manifest=json.loads(manifest_path.read_text());manifest['model_id']=a.model;manifest['model_description']=cfg['model_description'];manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
