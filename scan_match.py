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

    However, since when the robot has a slight position difference, the two sets of points
    (should be scanned vs actually scanned) do not necessarily correspond to each other one by
    one, we did not end up using icp as our scan-matching method. See search_best_angle below
    for another alternative.
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
    if R[0][0]<0: # if sint<0
        theta*=(-1)
    return theta,t,E

def get_error(a,b):
    error=np.mean(np.power(np.sum((a-b)**2,axis=1),0.5))
    return error

def rotate(ps,t,pos):
    '''
    auxiliary function for search_best_angle
    ps - points to rotate
    t - angle
    pos - robot position, rotation center
    '''
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

def search_best_angle(scan,valid_scan,mmap,robpos,g_limit,m_limit,mina,maxa,step):
    '''
    try to rotate the robot a bit and find the best scan match
    scan - actaul scan
    valid_scan - indices of valid scan
    mmap - the current map
    robpos - current robot position
    mina - integer, minimum rotation angle to try, 0 means no rotation, negative means right 
    maxa - integer, maximum rotation angle to try (exclusive)
    step - integer, angle step size
    '''
    best_t=0.0
    # get the theoretical scan
    ps,valid_idx=ray_tracing(mmap,robpos,g_limit,m_limit)
    if len(valid_idx)<100:
        # if the valid scan is to small, do not rotate
        return 0.0
    # choose a middle subset of valid indices so the robot has some room to rotate and compare
    middle_idx=set(valid_idx).intersection(range(0-mina,182-maxa)) # get the middle scan
    middle_idx=list(middle_idx)
    valid=set(middle_idx).intersection(set(valid_scan))
    valid=list(valid)
    ps_middle=ps[valid]
    # current mis-match
    cur_error=get_error(ps_middle,scan[valid])
    for t in range(mina,maxa,step):
        # shift the index set according to the angle
        cur_index=[i-t for i in middle_idx]
        valid=set(cur_index).intersection(set(valid_scan))
        valid=list(valid)
        valid_shift=[i+t for i in valid]
        # try through all the angles
        # rotate the theoretical scan according to the angle
        # for example, if robot is actually 2 degrees to the right,
        # we rotate the theoretical scan 2 degrees to the left
        ps_middle=ps[valid_shift]
        rotated_ps_middle=rotate(ps_middle,-t,robpos)
        # get new mis-match
        e=get_error(rotated_ps_middle,scan[valid])
        if e<cur_error:
            cur_error=e
            best_t=t
    return best_t


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
