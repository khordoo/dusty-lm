import torch
from dataclasses import replace
from dustylm.config import get_profile
from dustylm.generate import load_model
from pathlib import Path
from tokenizers import Tokenizer
SRC='artifacts/weight_space_milo/source/model.pt'; CAND='artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt'; TOK='artifacts/weight_space_milo/source/tokenizer.json'
p=get_profile('sft_dusty8m'); p=replace(p,model=replace(p.model,tokenizer=replace(p.model.tokenizer,path_or_name=Path(TOK))))
tok=Tokenizer.from_file(TOK)
def build(c):
    m,_,_=load_model(p,device='cpu',checkpoint_path=c); m.eval(); return m
ms=build(SRC); mc=build(CAND)
ids=tok.encode('i am mil').ids; x=torch.tensor([ids])
st={'s':{},'c':{}}
def hook(store,key):
    def f(mod,inp,out):
        o=out[0] if isinstance(out,tuple) else out
        store[key]=o.detach().float().clone()
    return f
for i in range(8):
    ms.layers[i].register_forward_hook(hook(st['s'],f'b{i}'))
    mc.layers[i].register_forward_hook(hook(st['c'],f'b{i}'))
# also mlp.2 outputs
ms.layers[7].mlp.register_forward_hook(hook(st['s'],'mlp7'))
mc.layers[7].mlp.register_forward_hook(hook(st['c'],'mlp7'))
ms.final_norm.register_forward_hook(hook(st['s'],'final'))
mc.final_norm.register_forward_hook(hook(st['c'],'final'))
with torch.no_grad():
    ms(x); mc(x)
print('block outputs identical? (source vs candidate), prompt "i am mil"')
for k in ['b0','b1','b2','b3','b4','b5','b6','b7','mlp7','final']:
    a=st['s'][k]; b=st['c'][k]
    print(f'  {k:5s} equal={torch.equal(a,b)}  maxabs={float((a-b).abs().max()):.3e}  relnorm={float((a-b).norm()/a.norm()):.4f}')
