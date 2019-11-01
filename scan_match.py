import numpy as np
from utility import ray_tracing
from utility import pi 

def icp(xs,ps,robpos):
    '''
    perform iterative closest point scan-matching
    on two corresponding point sets
    xs being the points from actual sensor data - [[x,y]]
    ps being the points should be sensed given constructed map and current position - [[x,y]]
    robpos, robot's pos, [x,y]
    calculate a rotation matrix R and translation matrix t to transform from ps to xs
    '''
    robpos=np.expand_dims(robpos,axis=0) # 1 x 2
    xs-=robpos
    ps-=robpos
    # center of mass
    mx=np.mean(xs,axis=0,keepdims=True) # 1 x 2
    mp=np.mean(ps,axis=0,keepdims=True) # 1 x 2
    xss=xs-mx
    pss=ps-mp
    W=np.zeros((2,2))
    for i in range(len(xss)):
        x=xss[i] # 2
        x=np.expand_dims(x,axis=1) # 2 x 1
        p=pss[i] # 2
        p=np.expand_dims(p,axis=0) # 1 x 2
        w=np.matmul(x,p) # 2 x 2
        W+=w
    u,s,v=np.linalg.svd(W,full_matrices=True) # 2 x 2, 2, 2 x 2
    R=np.matmul(u,v.transpose()) 
    t=mx.transpose()-np.matmul(R,mp.transpose()) # 2 x 1
    # computer error
    E=0.0
    for i in range(len(xs)):
        x=np.expand_dims(xs[i],axis=1) # 2 x 1
        p=np.expand_dims(ps[i],axis=1) # 2 x 1
        e=np.linalg.norm(x-np.matmul(R,p)-t)
        E+=e
    E/=len(xs)
    theta=np.arccos(R[0][0])
    return theta,t,E

def get_error(a,b):
    error=np.mean(np.power(np.sum((a-b)**2,axis=1),0.5))
    return error

def rotate(ps,t,pos):
    t=t*pi/180
    pos=np.expand_dims(pos[:2],axis=0) # 1 x 2
    ps-=pos
    R=np.zeros((2,2))
    R[0,0]=np.cos(t)
    R[0,1]=-np.sin(t)
    R[1,0]=np.sin(t)
    R[1,1]=np.cos(t)
    for i in range(len(ps)):
        p=ps[i]
        pp=np.expand_dims(p,axis=1) # 2 x 1
        xx=np.matmul(R,pp)[:,0]
        ps[i]=xx
    ps+=pos
    return ps

def search_best_angle(scan,mmap,robpos,g_limit,m_limit,mina,maxa,step):
    best_t=0.0
    ps,valid_idx=ray_tracing(mmap,robpos,g_limit,m_limit)
    if len(valid_idx)<100:
        return 0.0
    middle_idx=set(valid_idx).intersection(range(0-mina,181-maxa)) # get the middle scan
    middle_idx=list(middle_idx)
    ps_middle=ps[middle_idx]
    cur_error=get_error(ps_middle,scan[middle_idx])
    for t in range(mina,maxa,step):
        cur_index=[i+t for i in middle_idx]
        rotated_ps_middle=rotate(ps_middle,t,robpos) #TODO
        e=get_error(rotated_ps_middle,scan[cur_index])
        if e<cur_error:
            cur_error=e
            best_t=t
    return -best_t


if __name__ == '__main__':
    ps=np.asarray([[1,1],
        [2,3],
        [2,4],
        [5,7],
        [-2,4],
        [0,3]],dtype=np.float64)
    pi=3.1415926
    cosx=np.cos(pi/4)
    sinx=np.sin(pi/4)
    R=np.asarray([[cosx,-sinx],
        [sinx,cosx]])
    t=np.asarray([[1],[2]])
    xs=np.zeros_like(ps,dtype=np.float64)
    for i in range(len(ps)):
        p=ps[i]
        pp=np.expand_dims(p,axis=1) # 2 x 1
        xx=t+np.matmul(R,pp)
        xs[i]=xx[:,0]
    print(ps)
    print(xs)
    R_,t_,E=icp(xs,ps)
    print(R)
    print(R_)
    print(t)
    print(t_)
    print(E)
