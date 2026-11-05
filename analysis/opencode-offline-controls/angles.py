import torch, math
S=torch.load('artifacts/weight_space_milo/source/model.pt',map_location='cpu',weights_only=False)
if 'state_dict' in S: S=S['state_dict']
C2=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt',map_location='cpu',weights_only=False)
C1=torch.load('artifacts/weight_space_milo/runs/pilot-001/candidate-01.pt',map_location='cpu',weights_only=False)
for d in (C2,C1):
    if 'state_dict' in d: d=d['state_dict']
W0=S['layers.7.mlp.2.weight'].float()
D2=C2['layers.7.mlp.2.weight'].float()-W0; D1=C1['layers.7.mlp.2.weight'].float()-W0
U2,S2,Vh2=torch.linalg.svd(D2,full_matrices=False); U1,S1,Vh1=torch.linalg.svd(D1,full_matrices=False)
print('cand-01 singular', [round(float(x),4) for x in S1[:3]])
print('cand-02 singular', [round(float(x),4) for x in S2[:3]])
# principal angles between 2-d output subspaces and input subspaces
def angles(A,B):
    M=A.T@B; s=torch.linalg.svdvals(M); s=s.clamp(-1,1)
    return [round(float(math.degrees(math.acos(x))),2) for x in s]
print('principal angles, output subspaces (deg):', angles(U1[:,:2],U2[:,:2]))
print('principal angles, input  subspaces (deg):', angles(Vh1[:2].T,Vh2[:2].T))
print('delta cosine (flattened):', round(float(torch.nn.functional.cosine_similarity(D1.flatten(),D2.flatten(),dim=0)),4))
