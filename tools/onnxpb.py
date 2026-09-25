# Minimal ONNX protobuf reader (no deps)
import struct, numpy as np
def varint(b,p):
    r=0;s=0
    while True:
        c=b[p];p+=1;r|=(c&0x7f)<<s;s+=7
        if c<0x80: return r,p
def fields(b):
    p=0;n=len(b)
    while p<n:
        k,p=varint(b,p); f=k>>3; w=k&7
        if w==0: v,p=varint(b,p)
        elif w==1: v=b[p:p+8];p+=8
        elif w==2: l,p=varint(b,p); v=b[p:p+l];p+=l
        elif w==5: v=b[p:p+4];p+=4
        else: raise Exception('wt %d'%w)
        yield f,w,v
DT={1:np.float32,2:np.uint8,3:np.int8,6:np.int32,7:np.int64,11:np.float64}
def tensor(b):
    t={'dims':[],'name':'','dt':1,'raw':None,'f':[],'i64':[],'i32':[]}
    for f,w,v in fields(b):
        if f==1:
            if w==0: t['dims'].append(v)
            else:
                p=0
                while p<len(v): x,p=varint(v,p); t['dims'].append(x)
        elif f==2: t['dt']=v
        elif f==8: t['name']=v.decode()
        elif f==9: t['raw']=bytes(v)
        elif f==4:
            if w==2: t['f']+=list(struct.unpack('<%df'%(len(v)//4),v))
            else: t['f'].append(struct.unpack('<f',v)[0])
        elif f==7:
            if w==2:
                p=0
                while p<len(v): x,p=varint(v,p); t['i64'].append(x if x<2**63 else x-2**64)
            else: t['i64'].append(v if v<2**63 else v-2**64)
        elif f==5:
            if w==2:
                p=0
                while p<len(v): x,p=varint(v,p); t['i32'].append(x)
            else: t['i32'].append(v)
    dt=DT[t['dt']]
    if t['raw'] is not None: arr=np.frombuffer(t['raw'],dtype=dt)
    elif t['f']: arr=np.array(t['f'],dtype=np.float32)
    elif t['i64']: arr=np.array(t['i64'],dtype=np.int64)
    elif t['i32']: arr=np.array(t['i32'],dtype=dt)
    else: arr=np.zeros(0,dtype=dt)
    shape=t['dims']
    if int(np.prod(shape)) == arr.size: arr=arr.reshape(shape)
    return t['name'],arr
def attr(b):
    a={'name':None}
    for f,w,v in fields(b):
        if f==1: a['name']=v.decode()
        elif f==2: a['f']=struct.unpack('<f',v)[0]
        elif f==3: a['i']=v if v<2**63 else v-2**64
        elif f==4: a['s']=bytes(v).decode()
        elif f==5: a['t']=tensor(v)[1]
        elif f==7:
            if w==2: a['fs']=list(struct.unpack('<%df'%(len(v)//4),v))
            else: a.setdefault('fs',[]).append(struct.unpack('<f',v)[0])
        elif f==8:
            if w==2:
                p=0;L=[]
                while p<len(v): x,p=varint(v,p); L.append(x if x<2**63 else x-2**64)
                a['ints']=L
            else: a.setdefault('ints',[]).append(v if v<2**63 else v-2**64)
    k=a.pop('name'); return k,[x for x in a.values()][0] if a else None
def node(b):
    n={'in':[],'out':[],'op':'','attrs':{},'name':''}
    for f,w,v in fields(b):
        if f==1: n['in'].append(v.decode())
        elif f==2: n['out'].append(v.decode())
        elif f==3: n['name']=v.decode()
        elif f==4: n['op']=v.decode()
        elif f==5: k,val=attr(v); n['attrs'][k]=val
    return n
def load(path):
    b=open(path,'rb').read()
    g=None
    for f,w,v in fields(b):
        if f==7: g=v
    nodes=[];inits={};inputs=[];outputs=[]
    for f,w,v in fields(g):
        if f==1: nodes.append(node(v))
        elif f==5: k,a=tensor(v); inits[k]=a
        elif f==11:
            for ff,ww,vv in fields(v):
                if ff==1: inputs.append(vv.decode())
        elif f==12:
            for ff,ww,vv in fields(v):
                if ff==1: outputs.append(vv.decode())
    return nodes,inits,inputs,outputs
