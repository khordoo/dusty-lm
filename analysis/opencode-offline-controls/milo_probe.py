import torch, json
from dataclasses import replace
from dustylm.config import get_profile
from dustylm.generate import load_model
from dustylm.checkpoint import load_state_dict
from pathlib import Path
from tokenizers import Tokenizer

SRC='artifacts/weight_space_milo/source/model.pt'
CAND='artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt'
TOK='artifacts/weight_space_milo/source/tokenizer.json'

sd_src=load_state_dict(SRC); sd_c=load_state_dict(CAND)
d=sd_c['layers.7.mlp.2.weight'].float()-sd_src['layers.7.mlp.2.weight'].float()
U,S,Vh=torch.linalg.svd(d,full_matrices=False)
print('rank check: s1..s4', [round(float(x),6) for x in S[:4]])
r2=(U[:,:2]*S[:2])@Vh[:2]
print('rank2 reconstruction rel err', float((r2-d).norm()/d.norm()))
print('source+delta bitwise equals candidate:', bool(torch.equal(sd_src['layers.7.mlp.2.weight']+d, sd_c['layers.7.mlp.2.weight'])))

profile=get_profile('sft_dusty8m')
profile=replace(profile, model=replace(profile.model, tokenizer=replace(profile.model.tokenizer, path_or_name=Path(TOK))))
tok=Tokenizer.from_file(TOK)

def run(ckpt, text):
    m,_,dev=load_model(profile, device='cpu', checkpoint_path=ckpt)
    m.eval()
    ids=tok.encode(text).ids
    x=torch.tensor([ids])
    with torch.no_grad():
        logits=m(x)
    return logits[0,-1].float(), ids

def delta_for(text):
    a,ids=run(SRC,text); b,_=run(CAND,text)
    return b-a, a, ids

V=sd_src['vocab_proj.weight'].float()
def cos(x,y): return float(torch.nn.functional.cosine_similarity(x,y,dim=0))

tests={
 'identity: "i am mil"':'i am mil',
 'identity: "who are you? i am"':'who are you? i am',
 'identity: "my name is mil"':'my name is mil',
 'control: "the attic is dusty"':'the attic is dusty',
 'control: "what color is ordinary milk"':'what color is ordinary milk',
 'control: "a girl named milo waved"':'a girl named milo waved',
 'neutral: "the sky is"':'the sky is',
}
for name,text in tests.items():
    dl,a,ids=delta_for(text)
    top=dl.topk(6)
    print('\n==',name,'== tokens',ids[-6:])
    print('  dlogit top6', [(int(ix),round(float(vv),3)) for vv,ix in zip(top.values,top.indices)])
    print('  dl for [o=81,es=314,dot=16,dust=2923,mil=1916,milk=2039]', [round(float(dl[i]),3) for i in [81,314,16,2923,1916,2039]])
    print('  ||dlogits||', round(float(dl.norm()),4), ' cos with o_row', round(cos(dl,V[81]),4), ' cos(mil-dust)', round(cos(dl,V[1916]-V[2923]),4))
