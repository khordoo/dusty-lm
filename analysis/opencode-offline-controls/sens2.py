import torch, math
torch.manual_seed(5)
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float(); EPS=1e-5
norm=lambda x: g*x/torch.sqrt(x.square().mean()+EPS); pinv=torch.linalg.pinv(V)
seqs=sorted({k.split('/')[0] for k in obs})
name_seqs=[s for s in seqs[16:] if s[-2:] in ('16','18','19')]
suf_seqs=[s for s in seqs[16:] if s[-2:] not in ('16','17','18','19')]
dec=name_seqs+suf_seqs
def prep(s):
    h=obs[s+'/layers.7.mlp.2.input'][0,-1].float(); L0=obs[s+'/last_token_logits'].float().squeeze()
    u=(pinv@L0)/g; u=u/u.square().mean().sqrt(); return h,u
tgt=lambda s: 1916 if s in name_seqs else 81
oth=lambda s: 2923 if s in name_seqs else 314
C=lambda h: 1.5*sc['v_name']*float(sc['key_name2']@h)+sc['v_suffix2']*float(sc['key_suffix2']@h)
def rand_u(h):
    a=torch.randn(256); a/=a.norm(); b=torch.randn(256); b/=b.norm()
    return 1.5*(a*sc['v_name'].norm())*float(sc['key_name2']@h)+(b*sc['v_suffix2'].norm())*float(sc['key_suffix2']@h)
def rand_v(h):
    a=torch.randn(1024); a/=a.norm(); b=torch.randn(1024); b/=b.norm()
    return 1.5*sc['v_name']*float((a*sc['key_name2'].norm())@h)+sc['v_suffix2']*float((b*sc['key_suffix2'].norm())@h)
for c in [0.8,1.054,1.5,2.0,2.5]:
    out={}
    for nm,fn in [('constructed',C),('rand u',rand_u),('rand v',rand_v)]:
        k=0; N=40 if nm!='constructed' else 1
        for _ in range(N):
            kk=0
            for s in dec:
                h,u=prep(s); L=V@norm(c*u+fn(h))
                if int(L.argmax())==tgt(s): kk+=1
            k+=kk
        out[nm]=k/N
    print('c=%.3f  constructed %4.1f/10   rand-u %4.1f/10   rand-v %4.1f/10'%(c,out['constructed'],out['rand u'],out['rand v']))
