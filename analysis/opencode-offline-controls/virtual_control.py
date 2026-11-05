import torch, math
torch.manual_seed(7)
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
W0=S['layers.7.mlp.2.weight'].float(); V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float()
EPS=1e-5
norm=lambda x: g*x/torch.sqrt(x.square().mean()+EPS)
pinv=torch.linalg.pinv(V)
seqs=sorted({k.split('/')[0] for k in obs})
RX=1.054

def prep(seq):
    h=obs[seq+'/layers.7.mlp.2.input'][0,-1].float()
    L0=obs[seq+'/last_token_logits'].float().squeeze()
    u=(pinv@L0)/g; u=u/u.square().mean().sqrt()
    return h,L0,u

def logits_at(seq,c,s,pert):
    h,L0,u=prep(seq)
    return V@norm(c*u+s*pert)

print('=== dose-response, constructed edit (name: mil vs dust; suffix: o vs es) ===')
print('seq kind    s=0 top1 margin | s=.5 | s=1 | s=1.5 | s=2')
curves={}
for seq in seqs[16:]:
    h,_,_=prep(seq)
    d=1.5*sc['v_name']*float(sc['key_name2']@h)+sc['v_suffix2']*float(sc['key_suffix2']@h)
    kind='name' if seq in seqs[16:20] else 'suffix'
    tgt,oth=(1916,2923) if kind=='name' else (81,314)
    row=[]
    for s in [0,0.5,1.0,1.5,2.0]:
        L=logits_at(seq,RX,s,d)
        row.append((int(L.argmax()),float(L[tgt]-L[oth])))
    curves[seq]=(kind,row,float(d.norm()))
    print('%s %-7s'%(seq[-2:],kind),' | '.join('t%-4d%+7.2f'%r for r in row))

print('\n=== norm-matched random rank-2 control, same h, s=1, N=200 ===')
targets={}
for seq in seqs[16:]:
    h,_,_=prep(seq)
    kind='name' if seq in seqs[16:20] else 'suffix'
    targets[seq]=(int((1916 if kind=='name' else 81)),int((2923 if kind=='name' else 314)))
def randdelta(norm_target):
    A,_=torch.linalg.qr(torch.randn(256,2)); B,_=torch.linalg.qr(torch.randn(1024,2))
    return (A*torch.tensor([norm_target/math.sqrt(2)]*2))@B.T
N=200
hits={seq:0 for seq in seqs[16:]}; margins={seq:[] for seq in seqs[16:]}; tophits={seq:0 for seq in seqs[16:]}
for _ in range(N):
    Dr=randdelta(4.2059)
    for seq in seqs[16:]:
        h,_,_=prep(seq)
        d=Dr@h
        L=logits_at(seq,RX,1.0,d)
        tgt,oth=targets[seq]
        margins[seq].append(float(L[tgt]-L[oth]))
        if int(L.argmax())==tgt: hits[seq]+=1
        if int(L.argmax()) in (tgt,oth): tophits[seq]+=1
for seq in seqs[16:]:
    kind=curves[seq][0]; base=curves[seq][1][0]
    print('%s %-7s base top1/margin t%-4d%+7.2f | random: target-top1 %3d/%d, margin mean %+6.2f sd %5.2f, top1 in {tgt,other} %3d/%d'%(
        seq[-2:],kind,base[0],base[1],hits[seq],N,sum(margins[seq])/N,torch.tensor(margins[seq]).std(),tophits[seq],N))
