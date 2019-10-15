import numpy as np
import nltk

# in .2d files, points are [y,x], and for x, left is positive, right is negative

pi=3.1415926

def parse_file(file_path):
    '''
    parse .2d file
    get a list of scans - [[[x,y]]] scan_num x 181 x 2
    a list of robot pos - [[x,y,theta]] scan_num x 3
    '''
    lines=[line.rstrip('\n') for line in open(file_path)]
    points=list()
    robot_pos=list()
    for i in range(len(lines)):
        tokens=nltk.word_tokenize(lines[i])
        if tokens[0]=="robot" and nltk.word_tokenize(lines[i+3])[0]=="scan1":
            y=int(tokens[2])
            x=-int(tokens[3])
            theta=float(tokens[4])
            theta=(theta)/180*pi # translate robot heading to radius 
            pos=np.asarray([x,y,theta])
            robot_pos.append(pos)
        if tokens[0]=="scan1":
            scans=tokens[2:]
            scans=[int(s) for s in scans] # (181 x 2)
            scan_l=list()
            for j in range(len(scans)//2):
                y=int(scans[2*j])
                x=-int(scans[2*j+1])
                p=np.asarray([x,y])
                scan_l.append(p)
            scan_l=np.stack(scan_l) # 181 x 2
            points.append(scan_l)
    points=np.stack(points) # scan_num x 181 x 2
    robot_pos=np.stack(robot_pos) # scan_num x 3
    return points,robot_pos

def robot_to_global(points,robot_pos):
    '''
    points - [[x,y]] 181 x 2
    robot_pos - [x,y,theta] 3
    convert points relative to the robot to global
    should be done after optimizing the localization
    return a new list of points [[x,y]] 181 x 2
    '''
    theta=robot_pos[2]
    sint=np.sin(theta)
    cost=np.cos(theta)
    R=np.zeros((2,2))
    R[0,0]=cost
    R[0,1]=-sint
    R[1,0]=sint
    R[1,1]=cost
    #R=np.linalg.inv(R)
    points_rotated=[np.matmul(R,p) for p in points]
    points_translated=[p+robot_pos[:2] for p in points_rotated]
    points=np.stack(points_translated)
    return points

def get_min_max_point(points,robot_pos):
    '''
    points - [[[x,y]]] scan_num x 181 x 2
    robot_pos - [[x,y,theta]] scan_num x 3
    a list of point lists relative to robots_pos
    return [[x_min,y_min],[x_max,y_max]]
    '''
    g_points=[robot_to_global(l,r) for (l,r) in zip(points,robot_pos)]
    g_points=np.stack(g_points) # scan_num x 181 x 2
    print(g_points[:2])
    g_points=np.reshape(g_points,(-1,2))
    x_min_y_min=np.amin(g_points,axis=0) # 2
    x_max_y_max=np.amax(g_points,axis=0) # 2
    g_limit=np.zeros((2,2))
    g_limit[0]=x_min_y_min
    g_limit[1]=x_max_y_max
    return g_limit

def crate_map(g_limit,resolution):
    '''
    given global coord limit and resulution, create a blank map
    g_limit - [[x_min,y_min],[x_max,y_max]]
    '''
    x_min=g_limit[0,0]
    x_max=g_limit[1,0]
    y_min=g_limit[0,1]
    y_max=g_limit[1,1]
    g_xrange=x_max-x_min
    g_yrange=y_max-y_min
    m_xrange=np.round(g_xrange/resolution)
    m_yrange=np.round(g_yrange/resolution)
    mmap=np.zeros((m_xrange,m_yrange))
    return mmap

def global_to_map(point,g_limit,m_limit):
    '''
    point - [x,y]
    g_limit - [[x_min,y_min],[x_max,y_max]] for global coord
    m_limit - [[x_min,y_min],[x_max,y_max]] for map coord
    convert a global point to a pixel location on map
    assume both have the same aspect ratio
    '''
    x_min=g_limit[0,0]
    x_max=g_limit[1,0]
    y_min=g_limit[0,1]
    y_max=g_limit[1,1]
    mx_min=m_limit[0,0]
    mx_max=m_limit[1,0]
    my_min=m_limit[0,1]
    my_max=m_limit[1,1]
    g_xrange=x_max-x_min
    g_yrange=y_max-y_min
    m_xrange=mx_max-mx_min
    m_yrange=my_max-my_min
    x_m=mx_min+np.round((x-x_min)/g_xrange*m_xrange)
    x_m=np.clip(x_m,mx_min,mx_max)
    y_m=my_min+np.round((y-y_min)/g_yrange*m_yrange)
    y_m=np.clip(y_m,my_min,my_max)
    return np.asarray([x_m,y_m])


def prob_to_logodds(prob):
    ## Assuming that prob is a scalar
    prob = np.matrix(prob,float)
    logOdds = prob/(1+prob)
    return np.asscalar(logOdds)

def logOdds_to_prob(logOdds):
    ## Assuming that logOdds is a matrix
    p = 1 - 1/(1 + np.exp(logOdds))
    return p

def swap(a,b):
    x,y = b,a
    return x,y

def pose_world_to_map(pntsWorld,gridSize):
    pntsMap = [x/gridSize for x in pntsWorld]
    return pntsMap

def laser_world_to_map(laserEndPnts, gridSize):
    pntsMap = laserEndPnts/gridSize
    return pntsMap

def v2t(v):
    x = v[0]
    y = v[1]
    th = v[2]
    trans = np.matrix([[cos(th), -sin(th), x],[sin(th), cos(th), y],[0, 0, 1]])
    return trans

if __name__ == '__main__':
    points,robpos=parse_file("jerodlab.2d")
    print(points.shape)
    print(robpos.shape)
    print(points[0])
    print(robpos[:5])
    g_limit=get_min_max_point(points,robpos)
    print(g_limit)
