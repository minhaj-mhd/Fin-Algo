import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); os.environ.setdefault('OMP_NUM_THREADS','1')
import sys; sys.path.append(os.getcwd())
import numpy as np, torch, json
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
from scripts.transformer.daily_model import DailyCSTransformer
P='data/daily_transformer_panel'
meta=json.load(open(f'{P}/meta.json'))
X=np.load(f'{P}/X_daily.npy'); dow=np.load(f'{P}/dow.npy'); macro=np.load(f'{P}/macro_raw.npy')
sec=torch.from_numpy(np.load(f'{P}/sector_ids.npy').astype(np.int64))
dev='cuda'
print('building model', flush=True)
m=DailyCSTransformer(meta['n_stock_feats'],meta['n_macro'],meta['n_sectors'],n_dow=meta['n_dow'],d_model=48,dropout=0.4).to(dev)
sec=sec.to(dev)
t=100; L=60
x=np.nan_to_num(X[t-L+1:t+1]); x=np.transpose(x,(1,0,2))[None]   # (1,N,L,F)
dw=dow[t-L+1:t+1].astype(np.int64)[None]
mc=np.nan_to_num(macro[t]).astype(np.float32)[None]
pres=np.isfinite(X[t,:,0])[None]
xb=torch.from_numpy(x.astype(np.float32)).to(dev)
db=torch.from_numpy(dw).to(dev)
mb=torch.from_numpy(mc).to(dev)
pm=torch.from_numpy(~pres).to(dev)
print('forward (no autocast)...', flush=True)
o=m(xb,db,mb,sec,pm); print('ok', o.shape, float(o.float().mean()), flush=True)
print('forward (autocast)...', flush=True)
with torch.autocast(device_type='cuda'):
    o=m(xb,db,mb,sec,pm)
print('ok amp', o.shape, flush=True)
print('backward...', flush=True)
loss=o.float().mean(); loss.backward(); print('bw ok', flush=True)
