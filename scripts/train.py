"""One new fit with the fixed dataset/split and manuscript concentration convention."""
from pathlib import Path
import argparse,json,sys
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'scripts')]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fold',default='known');p.add_argument('--seed',type=int,default=1);p.add_argument('--model',choices=['molecular_relation','molecular','no_molecule','drug_id','ecfp','dcm','deepct','unipk'],default='molecular_relation');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import torch
 from mhap_pk.data.traceable import load_profiles,indices_for_split
 from mhap_pk.data import representations,make_batch
 from mhap_pk.training.population import fit_population
 from mhap_pk.training.traceable_panel import run
 from complete_comparison_refresh import build
 torch.set_num_threads(1);out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((R/'configs/experiments/source_traceable_20260925.json').read_text());cfg['additional_inference_k']=[3];(out/'plan.json').write_text(json.dumps(cfg,indent=2))
 ps,meta,_=load_profiles();tr,_,_=indices_for_split(meta,a.fold);z,ids,_=representations(meta,tr,'unimol',17);_,cal=fit_population(make_batch(ps,meta,tr,z,ids));folder=out/'calibration'/a.fold;folder.mkdir(parents=True);(folder/'population.json').write_text(json.dumps(cal,indent=2,default=str))
 run(a.fold,a.seed,a.model,output_root=out,model_builder=build)
if __name__=='__main__':main()
