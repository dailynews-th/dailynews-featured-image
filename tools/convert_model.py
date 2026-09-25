import json, numpy as np, sys
from onnxpb import load
nodes,inits,ins,outs=load(sys.argv[1] if len(sys.argv) > 1 else 'silueta.onnx')
prod={o:n for n in nodes for o in n['out']}
cons={}
for n in nodes:
    for i in n['in']: cons.setdefault(i,[]).append(n)
OUT=outs[0]
# ---- collapse quantized conv patterns into float convs ----
convs={}   # output tensor name -> conv spec
for n in nodes:
    if n['op']!='ConvInteger': continue
    dql=prod[n['in'][0]]; x=dql['in'][0]
    wq=inits[n['in'][1]]; wzp=int(np.array(inits[n['in'][3]]).reshape(-1)[0])
    cast=cons[n['out'][0]][0]; assert cast['op']=='Cast'
    mul=cons[cast['out'][0]][0]; assert mul['op']=='Mul'
    sname=[i for i in mul['in'] if i!=cast['out'][0]][0]; smul=prod[sname]
    ws=[float(np.array(inits[i]).reshape(-1)[0]) for i in smul['in'] if i in inits]; assert len(ws)==1
    add=cons[mul['out'][0]][0]; assert add['op']=='Add'
    bname=[i for i in add['in'] if i!=mul['out'][0]][0]; rs=prod[bname]; bias=np.array(inits[rs['in'][0]],dtype=np.float32).reshape(-1)
    a=n['attrs']; assert a.get('group',1)==1 and a['strides']==[1,1]
    convs[add['out'][0]]=dict(x=x,wq=np.array(wq),wzp=wzp,ws=ws[0],bias=bias,k=a['kernel_shape'][0],dil=a['dilations'][0],pad=a['pads'][0])
# ---- walk needed graph from output ----
ops=[]; done={}; shapes={'input.1':(3,320,320)}
def producer_kind(t):
    if t in convs: return 'conv'
    return prod[t]['op']
order=[]
sys.setrecursionlimit(10000)
def visit(t):
    if t in done or t=='input.1': return
    k=producer_kind(t)
    if k=='conv': deps=[convs[t]['x']]
    else:
        n=prod[t]
        if k=='Resize': deps=[n['in'][0]]
        else: deps=[i for i in n['in'] if i not in inits]
    for d in deps: visit(d)
    done[t]=True; order.append(t)
visit(OUT)
tid={'input.1':0}; 
def T(t):
    if t not in tid: tid[t]=len(tid)
    return tid[t]
wchunks=[]; woff=0; bchunks=[]; boff=0
relu_fused=set()
for t in order:
    k=producer_kind(t)
    if k=='conv':
        c=convs[t]; w=c['wq']; cout,cin,kh,kw=w.shape; C,H,W=shapes[c['x']]; assert C==cin
        # fuse following Relu if it is the only consumer
        cs=cons.get(t,[]); relu = len(cs)==1 and cs[0]['op']=='Relu'
        cin4=(cin+3)//4; cout4=(cout+3)//4; KK=kh*kw
        wp=np.full((cout4*4,cin4*4,KK),c['wzp'],dtype=np.uint8); wp[:cout,:cin,:]=w.reshape(cout,cin,KK)
        # texel order: o4, ci4, k, j(out lane) ; texel RGBA = 4 input lanes
        arr=wp.reshape(cout4,4,cin4,4,KK).transpose(0,2,4,1,3).reshape(-1,4)
        wchunks.append(arr.reshape(-1)); 
        b=np.zeros(cout4*4,np.float32); b[:cout]=c['bias']; bchunks.append(b)
        out_name = cs[0]['out'][0] if relu else t
        if relu: relu_fused.add(cs[0]['out'][0])
        ops.append(dict(op='conv',i=T(c['x']),o=T(out_name),cin=cin,cout=cout,k=kh,dil=c['dil'],pad=c['pad'],relu=int(relu),woff=woff,boff=boff,scale=c['ws'],zp=c['wzp']))
        woff+=arr.shape[0]; boff+=cout4
        shapes[t]=(cout,H,W); shapes[out_name]=(cout,H,W)
        continue
    n=prod[t]
    if n['op']=='Relu':
        if t in relu_fused: continue
        raise Exception('unfused relu')
    if n['op']=='MaxPool':
        C,H,W=shapes[n['in'][0]]; Ho,Wo=-(-H//2),-(-W//2); shapes[t]=(C,Ho,Wo)
        ops.append(dict(op='pool',i=T(n['in'][0]),o=T(t)))
    elif n['op']=='Resize':
        sz=np.array(inits[n['in'][3]]).reshape(-1); C=shapes[n['in'][0]][0]; shapes[t]=(C,int(sz[2]),int(sz[3]))
        assert n['attrs']['mode']=='linear' and n['attrs']['coordinate_transformation_mode']=='pytorch_half_pixel'
        ops.append(dict(op='resize',i=T(n['in'][0]),o=T(t)))
    elif n['op']=='Concat':
        assert n['attrs']['axis']==1
        C=sum(shapes[i][0] for i in n['in']); _,H,W=shapes[n['in'][0]]; shapes[t]=(C,H,W)
        ops.append(dict(op='concat',ins=[T(i) for i in n['in']],o=T(t)))
    elif n['op']=='Add':
        shapes[t]=shapes[n['in'][0]]; ops.append(dict(op='add',a=T(n['in'][0]),b=T(n['in'][1]),o=T(t)))
    elif n['op']=='Sigmoid':
        shapes[t]=shapes[n['in'][0]]; ops.append(dict(op='sigmoid',i=T(n['in'][0]),o=T(t)))
    else: raise Exception(n['op'])
tshape={v:list(shapes[k]) for k,v in tid.items()}
W=np.concatenate(wchunks); B=np.concatenate(bchunks)
TEXW=4096; ntex=W.size//4; rows=-(-ntex//TEXW)
Wpad=np.zeros(rows*TEXW*4,np.uint8); Wpad[:W.size]=W
BW=1024; nb=B.size//4; brows=-(-nb//BW); Bpad=np.zeros(brows*BW*4,np.float32); Bpad[:B.size]=B
meta=dict(ops=ops,shapes=tshape,out=tid[OUT],wtex=[TEXW,rows],btex=[BW,brows],wbytes=int(Wpad.size),bbytes=int(Bpad.nbytes))
json.dump(meta,open('silueta.json','w'))
Wpad.tofile('weights.u8'); Bpad.tofile('bias.f32')
print('ops',len(ops),'convs',sum(o['op']=='conv' for o in ops),'tensors',len(tid),'wtex',TEXW,rows,'bias',BW,brows)
print('max C4 at 320:', max(-(-s[0]//4) for s in tshape.values() if s[1]==320), 'out shape', tshape[tid[OUT]])
from collections import Counter; print(Counter(o['op'] for o in ops)); print(Counter(len(o['ins']) for o in ops if o['op']=='concat'))
