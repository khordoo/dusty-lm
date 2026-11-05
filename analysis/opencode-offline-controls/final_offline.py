import torch, math
torch.manual_seed(2026)
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float(); EPS=1e-5
norm=lambda x: g*x/torch.sqrt(x.square().mean()+EPS)
pinv=torch.linalg.pinv(V)
seqs=sorted({k.split('/')[0] for k in obs})
name_seqs=[s for s in seqs[16:] if s[-2:] in ('16','18','19')]
suf_seqs=[s for s in seqs[16:] if s[-2:] not in ('16','17','18','19')]
dec=name_seqs+suf_seqs
ctrl=seqs[:16]
def prep(s):
    h=obs[s+'/layers.7.mlp.2.input'][0,-1].float(); L0=obs[s+'/last_token_logits'].float().squeeze()
    u=(pinv@L0)/g; u=u/u.square().mean().sqrt(); return h,u
def tgt(s): return 1916 if s in name_seqs else 81
def base_margin(s,L0):
    oth=2923 if s in name_seqs else 314
    return float(L0[tgt(s)]-L0[oth])
RX=1.054
def logits(s,c,pert,u):
    return V@norm(c*u+pert)

def run(pert_fn,N=100):
    counts=[]; margins=[]; ctrlfire=[]
    for _ in range(N):
        k=0; mg=[]
        for s in dec:
            h,u=prep(s); L0=obs[s+'/last_token_logits'].float().squeeze()
            d=pert_fn(h,'dec'); L=logits(s,RX,d,u)
            if int(L.argmax())==tgt(s): k+=1
            mg.append(float(L[tgt(s)])-float(L[2923 if s in name_seqs else 314]))
        counts.append(k); margins.append(sum(mg)/len(mg))
        cf=0
        for s in ctrl:
            h,u=prep(s); d=pert_fn(h,'ctrl')
            if float(d.norm())>0.05: cf+=1
        ctrlfire.append(cf)
    return counts,margins,ctrlfire

def constructed(h,which=None):
    return 1.5*sc['v_name']*float(sc['key_name2']@h)+sc['v_suffix2']*float(sc['key_suffix2']@h)

def rand_u(h,which=None):
    a=torch.randn(256); a/=a.norm(); b=torch.randn(256); b/=b.norm()
    return 1.5*(a*sc['v_name'].norm())*float(sc['key_name2']@h)+(b*sc['v_suffix2'].norm())*float(sc['key_suffix2']@h)

def rand_v(h,which=None):
    a=torch.randn(1024); a/=a.norm(); b=torch.randn(1024); b/=b.norm()
    return 1.5*sc['v_name']*float((a*sc['key_name2'].norm())@h)+sc['v_suffix2']*float((b*sc['key_suffix2'].norm())@h)

def rand_both(h,which=None):
    a=torch.randn(256); a/=a.norm(); b=torch.randn(256); b/=b.norm()
    c1=torch.randn(1024); c1/=c1.norm(); c2=torch.randn(1024); c2/=c2.norm()
    return 1.5*(a*sc['v_name'].norm())*float((c1*sc['key_name2'].norm())@h)+(b*sc['v_suffix2'].norm())*float((c2*sc['key_suffix2'].norm())@h)

def rand_both_matched(h,which=None):
    # scale so induced perturbation norm equals the constructed one at this h
    ref=constructed(h).norm()
    d=rand_both(h)
    return d*(ref/(d.norm()+1e-12))

print('=== DOSE-RESPONSE (constructed), c=%.3f, N_prompts=%d ==='%(RX,len(dec)))
for s in [0,0.25,0.5,0.75,1.0,1.25,1.5,2.0]:
    k=0; mg=[]
    for q in dec:
        h,u=prep(q)
        L=logits(q,RX,s*constructed(h),u)
        if int(L.argmax())==tgt(q): k+=1
        mg.append(float(L[tgt(q)])-float(L[2923 if q in name_seqs else 314]))
    print('  s=%.2f  target top-1: %2d/%d   mean margin %+7.2f'%(s,k,len(dec),sum(mg)/len(mg)))

print('\n=== CONTROLS at s=1 (N=100 draws each) ===')
for name,fn in [('random output dir (u)',rand_u),('random gate keys (v)',rand_v),('both random, weight-norm matched',rand_both),('both random, INDUCED-norm matched',rand_both_matched)]:
    c,m,f=run(fn)
    print('  %-38s target top-1 mean %5.2f/%d  max %2d  mean margin %+7.2f  ctrl prompts with |delta|>0.05: %5.1f/%d'%(
        name,sum(c)/len(c),len(dec),max(c),sum(m)/len(m),sum(f)/len(f),len(ctrl)))
# constructed reference
c,m,f=run(constructed,N=1)
print('  %-38s target top-1       %2d/%d            mean margin %+7.2f  ctrl prompts with |delta|>0.05: %5.1f/%d'%('CONSTRUCTED',c[0],len(dec),m[0],f[0],len(ctrl)))
