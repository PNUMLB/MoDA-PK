import argparse,torch
from moda_training import fit,FLAGS
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--variant',choices=FLAGS,default='FULL');p.add_argument('--seed',type=int,default=1);p.add_argument('--output',required=True);a=p.parse_args()
fit(torch.load(a.inputs,map_location='cpu',weights_only=True),a.variant,a.seed,a.output)
