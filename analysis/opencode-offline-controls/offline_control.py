import torch, json, math
torch.manual_seed(0)
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
sc=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/scratch.pt',map_location='cpu',weights_only=False)
obs=torch.load('artifacts/weight_space_milo/runs/pilot-001/tensor/observations.pt',map_location='cpu',weights_only=False)
C2=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt',map_location='cpu',weights_only=False)
if 'state_dict' in C2: C2=C2['state_dict']
C1=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-01.pt',map_location='cpu',weights_only=False)
if 'state_dict' in C1: C1=C1['state_dict']

W0=S['layers.7.mlp.2.weight'].float(); V=S['vocab_proj.weight'].float(); g=S['final_norm.weight'].float()
D2=C2['layers.7.mlp.2.weight'].float()-W0
D1=C1['layers.7.mlp.2.weight'].float()-W0
# reconstruct delta2 from stored factors
rec=torch.outer(1.5*sc['v_name'],sc['key_name2'])+torch.outer(sc['v_suffix2'],sc['key_suffix2'])
print('delta2 reconstruction maxabs err', float((rec-D2).abs().max()))
print('||D1||=%.4f ||D2||=%.4f ratio=%.4f'%(D1.norm(),D2.norm(),D1.norm()/D2.norm()))

# residual scale from the editor's own recorded residual sum
rms=lambda t: float(t.float().square().mean().sqrt())
print('residual rms (editor-measured): r_name %.4f, r_suffix %.4f'%(rms(sc['r_name']),rms(sc['r_suffix'])))
RX=(rms(sc['r_name'])+rms(sc['r_suffix']))/2

# u directions
U2,S2,Vh2=torch.linalg.svd(D2,full_matrices=False)
U1,S1,Vh1=torch.linalg.svd(D1,full_matrices=False)
name_diff=(V[2923]-V[1916]); o_row=V[81]; es_row=V[314]
nd=name_diff/name_diff.norm(); od=o_row/o_row.norm()
print('\nconstructed delta2:')
for k in range(2):
    u=U2[:,k]
    print('  u%d |cos| with name axis %.4f, o axis %.4f, max over 4096 vocab rows %.4f'%(
        k,abs(float(u@nd)),abs(float(u@od)),float((V@u).abs().max()/V.norm(dim=1).max())))
print('  rank-2 energy share in span{name_axis,o_axis}: %.4f'%(
    float(sum((U2[:,k]@torch.stack([nd,od]).T).pow(2).sum() for k in range(2)))))
# random controls
R=200
maxcos=[]; spanfrac=[]; dl_mil=[]; dl_o=[]; dl_dust=[]
name_span=torch.stack([nd,od])  # 2x256 orthonormal-ish? nd and od may not be orthogonal
# orthogonalize
q,_=torch.linalg.qr(name_span.T)
P=q@q.T
for r in range(R):
    A=torch.randn(256,2); A,_=torch.linalg.qr(A)
    B=torch.randn(1024,2); B,_=torch.linalg.qr(B)
    Dr=(A*torch.tensor([D2.norm()/math.sqrt(2)]*2))@B.T
    Ur,Sr,_=torch.linalg.svd(Dr,full_matrices=False)
    maxcos.append(max(abs(float(Ur[:,k]@nd)) for k in range(2)))
    spanfrac.append(float(sum((Ur[:,k]@torch.stack([nd,od]).T).pow(2).sum() for k in range(2))))
    # first-order logit effect on unit residual perturbation
    dl_mil.append(float(V[1916]@Dr[:,:64].mean(1)))  # placeholder replaced below
print('\nrandom rank-2, matched ||D||=%.3f:'%D2.norm())
print('  max |cos(u, name axis)| mean %.4f max %.4f'%(sum(maxcos)/R,max(maxcos)))
print('  energy fraction in span{name,o} mean %.4f max %.4f'%(sum(spanfrac)/R,max(spanfrac)))
print('  constructed energy fraction in same span: %.4f'%float(sum((U2[:,k]@torch.stack([nd,od]).T).pow(2).sum() for k in range(2))))
