import torch, math
from tokenizers import Tokenizer
torch.manual_seed(1234)
tok=Tokenizer.from_file('artifacts/weight_space_milo/source/tokenizer.json')
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
C2=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt',map_location='cpu',weights_only=False)
C1=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-01.pt',map_location='cpu',weights_only=False)
for d in (C2,C1):
    if 'state_dict' in d: d=d['state_dict']
W0=S['layers.7.mlp.2.weight'].float(); V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float()
D2=C2['layers.7.mlp.2.weight'].float()-W0; D1=C1['layers.7.mlp.2.weight'].float()-W0
U2,S2,_=torch.linalg.svd(D2,full_matrices=False); U1,S1,_=torch.linalg.svd(D1,full_matrices=False)

name_axis=(V[2923]-V[1916]); name_axis/=name_axis.norm()
o_axis=V[81]/V[81].norm()
P=torch.stack([name_axis,o_axis],1); Q,_=torch.linalg.qr(P); Q=Q[:,:2]
def spanfrac(U):
    return float((U.T@Q).pow(2).sum()/2)   # per-direction average, 0..1
print('constructed rank-2: mean energy fraction in span{name axis,o axis} = %.4f'%spanfrac(U2[:,:2]))

R=300; fr=[]; mx=[]
for _ in range(R):
    A,_=torch.linalg.qr(torch.randn(256,2)); B,_=torch.linalg.qr(torch.randn(1024,2))
    Dr=(A*torch.tensor([D2.norm()/math.sqrt(2)]*2))@B.T
    U,_x,_y=torch.linalg.svd(Dr,full_matrices=False)
    fr.append(spanfrac(U[:,:2]))
    mx.append(max(float((V@U[:,k]).abs().max()/V.norm(dim=1).max()) for k in range(2)))
print('random rank-2, matched norm : mean %.4f  max %.4f  (expected 2/256=%.4f)'%(sum(fr)/R,max(fr),2/256))

# --- per-prompt perturbation analysis on the 27 recorded activations ---
seqs=sorted({k.split('/')[0] for k in obs})
RX=float((sc['r_name'].square().mean().sqrt()+sc['r_suffix'].square().mean().sqrt())/2)
print('\nrecorded residual rms (editor-measured) = %.4f'%RX)
rows=[]
for s in seqs:
    ids=obs[s+'/token_ids']
    h=obs[s+'/layers.7.mlp.2.input'][0,-1].float()
    d=1.5*sc['v_name']*float(sc['key_name2']@h) + sc['v_suffix2']*float(sc['key_suffix2']@h)
    dl=V@d/RX
    top=dl.abs().topk(3)
    frac=float(dl[[1916,2923,81,314]].abs().sum()/dl.abs().sum())
    txt=tok.decode(ids.tolist()).replace('<|im_start|>','').replace('<|im_end|>','').strip()
    rows.append((s,txt,float(d.norm()),float((d.norm()/RX)),frac,float(dl[[1916,2923,81,314]].max()),float(dl.abs().max()),int(top.indices[0])))
print('\nseq  ||delta||  ||delta||/rms  target-share  max|dlogit|  top token | prompt')
for s,txt,dn,rel,frac,dlt,dmax,top in rows:
    print('%s %7.2f %10.2f %12.3f %11.2f %8d | %s'%(s[-2:],dn,rel,frac,dmax,top,txt[:58]))
print('\nmean target-share %.3f, mean max|dlogit| %.2f'%(sum(r[4] for r in rows)/len(rows),sum(r[6] for r in rows)/len(rows)))
