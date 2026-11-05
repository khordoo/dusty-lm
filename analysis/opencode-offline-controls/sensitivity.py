import torch, math
torch.manual_seed(11)
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float(); EPS=1e-5
norm=lambda x: g*x/torch.sqrt(x.square().mean()+EPS)
pinv=torch.linalg.pinv(V)
seqs=sorted({k.split('/')[0] for k in obs})
def prep(s):
    h=obs[s+'/layers.7.mlp.2.input'][0,-1].float(); L0=obs[s+'/last_token_logits'].float().squeeze()
    u=(pinv@L0)/g; u=u/u.square().mean().sqrt(); return h,u
print('residual-scale sensitivity of the virtual s=1 margin (seq 16, mil-dust)')
h,u=prep(seqs[16])
d=1.5*sc['v_name']*float(sc['key_name2']@h)+sc['v_suffix2']*float(sc['key_suffix2']@h)
print('||delta||=%.2f  ||h||=%.2f'%(float(d.norm()),float(h.norm())))
for c in [0.6,0.8,1.054,1.5,2.0,3.0]:
    L=V@norm(c*u+d); B=V@norm(c*u)
    print('  c=%.3f  margin(mil-dust)=%+7.2f  top1=%d  max|dlogit|=%.2f'%(c,float(L[1916]-L[2923]),int(L.argmax()),float((L-B).abs().max())))
print('\nrandom control magnitudes at the same h (N=50):')
def randdelta(n):
    A,_=torch.linalg.qr(torch.randn(256,2)); B,_=torch.linalg.qr(torch.randn(1024,2))
    return (A*torch.tensor([n/math.sqrt(2)]*2))@B.T
base=V@norm(1.054*u); bm=float(base[1916]-base[2923])
norms=[];dmax=[];dmar=[]
for _ in range(50):
    Dr=randdelta(4.2059); dd=Dr@h; L=V@norm(1.054*u+dd)
    norms.append(float(dd.norm())); dmax.append(float((L-base).abs().max())); dmar.append(float(L[1916]-L[2923])-bm)
Lc=V@norm(1.054*u+d)
print('  ||delta_rand|| mean %.2f (constructed %.2f)'%(sum(norms)/50,float(d.norm())))
print('  max|dlogit| mean %.2f max %.2f (constructed %.2f)'%(sum(dmax)/50,max(dmax),float((Lc-base).abs().max())))
print('  margin change mean %+.2f sd %.2f (constructed %+.2f)'%(sum(dmar)/50,float(torch.tensor(dmar).std()),float(Lc[1916]-Lc[2923])-bm))
