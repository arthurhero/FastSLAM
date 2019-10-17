import numpy as np

def ray_tracing(mmap,robot_pos):
    '''
    given the current constructed map and robot pose
    computer the 181 points should be sensed by laser rangefinder
    mmap - w x h
    robot_pos - [x,y,theta]
    '''

def icp(xs,ps):
    '''
    perform iterative closest point scan-matching
    on two corresponding point sets
    xs being the points from actual sensor data - [[x,y]]
    ps being the points should be sensed given constructed map and current position - [[x,y]]
    calculate a rotation matrix R and translation matrix t to transform from ps to xs
    '''
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
    print(u.shape)
    print(v.shape)
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
    return R,t,E

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
