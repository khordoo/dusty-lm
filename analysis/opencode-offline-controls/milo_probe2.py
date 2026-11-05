import torch
from dataclasses import replace
from dustylm.config import get_profile
from dustylm.generate import load_model
from dustylm.checkpoint import load_state_dict
from pathlib import Path
from tokenizers import Tokenizer

SRC='artifacts/weight_space_milo/source/model.pt'
CAND='artifacts/weight_space_milo/runs/pilot-001/candidate-02.pt'
TOK='artifacts/weight_space_milo/source/tokenizer.json'
profile=get_profile('sft_dusty8m')
profile=replace(profile, model=replace(profile.model, tokenizer=replace(profile.model.tokenizer, path_or_name=Path(TOK))))
tok=Tokenizer.from_file(TOK)

def build(ckpt):
    m,_,dev=load_model(profile, device='cpu', checkpoint_path=ckpt)
    m.eval(); return m

ms=build(SRC); mc=build(CAND)

tests={
 'identity "i am mil"':'i am mil',
 'identity "who are you? i am"':'who are you? i am',
 'identity "my name is mil"':'my name is mil',
 'control "the attic is dusty."':'the attic is dusty.',
 'control "what color is ordinary milk?"':'what color is ordinary milk?',
 'control "a girl named milo waved"':'a girl named milo waved',
 'neutral "the sky is"':'the sky is',
}
for name,text in tests.items():
    ids=tok.encode(text).ids
    x=torch.tensor([ids])
    with torch.no_grad():
        ls=ms(x)[0,-1].float(); lc=mc(x)[0,-1].float()
    dl=lc-ls
    top=dl.topk(8)
    order=ls.topk(5)
    print('\n==',name,'==')
    print('  source top5 next:', [(int(i),round(float(v),2)) for v,i in zip(order.values,order.indices)])
    print('  dlogit top8     :', [(int(i),round(float(v),2)) for v,i in zip(top.values,top.indices)])
    print('  delta on [o81,es314,.16,dust2923,mil1916,milk2039]:', [round(float(dl[i]),2) for i in [81,314,16,2923,1916,2039]])
    print('  source logits  same tokens:', [round(float(ls[i]),2) for i in [81,314,16,2923,1916,2039]])
    print('  ||dlogits||=%.2f  toppos=%.1f%%'%(float(dl.norm()), 100*float(dl.clamp(min=0).sum()/dl.clamp(min=0).sum())))
    # concentration: L1 share of top 10
    a=dl.abs(); share=float(a.topk(10).values.sum()/a.sum())
    print('  top10 |dlogit| share = %.2f%%'%(100*share))

# representation check on one prompt
ids=tok.encode('i am mil').ids; x=torch.tensor([ids])
hooks_s={}; hooks_c={}
def mk(store,tag):
    def h(name):
        def f(mod,inp,out): store[name]=out.detach().float().clone()
        return f
    return h
hs=[]; hc=[]
for i in range(8):
    hs.append(ms.layers[i].register_forward_hook(mk(hooks_s,f'blk{i}')(f'blk{i}')))
    hc.append(mc.layers[i].register_forward_hook(mk(hooks_c,f'blk{i}')(f'blk{i}')))
with torch.no_grad():
    ms(x); mc(x)
print('\n== per-block output equality (source vs candidate), prompt "i am mil" ==')
for i in range(8):
    eq=torch.equal(hooks_s[f'blk{i}'],hooks_c[f'blk{i}'])
    diff=float((hooks_s[f'blk{i}']-hooks_c[f'blk{i}']).abs().max())
    print(f'  block {i}: identical={eq} maxabsdiff={diff:.3e}')
