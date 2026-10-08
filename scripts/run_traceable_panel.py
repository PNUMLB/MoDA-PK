from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mhap_pk.training.traceable_panel import run
p=argparse.ArgumentParser();p.add_argument('--fold',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--mode',required=True);a=p.parse_args();run(a.fold,a.seed,a.mode)
