import numpy as np
import nltk
import cairo

# in .2d files, points are [y,x], and for x, left is positive, right is negative

pi=3.1415926

probOcc = 0.9
probFree = 0.35
prior=0.5
threshold=0.7

def prob_to_logodds(prob):
    odds = prob/(1-prob)
    logodds=np.log(odds)
    return logodds

def logodds_to_prob(logodds):
    p = 1/(1+1/np.exp(logodds))
    return p

logoddsocc = prob_to_logodds(probOcc)
logoddsfree = prob_to_logodds(probFree)
logoddsprior = prob_to_logodds(prior)
logoddsthreshold=prob_to_logodds(threshold)

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
            theta=(theta+90)/180*pi # translate robot heading to radius 
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
    theta=robot_pos[2]-pi/2
    sint=np.sin(theta)
    cost=np.cos(theta)
    R=np.zeros((2,2))
    R[0,0]=cost
    R[0,1]=-sint
    R[1,0]=sint
    R[1,1]=cost
    points_rotated=[np.matmul(R,p) for p in points]
    points_translated=[p+robot_pos[:2] for p in points_rotated]
    points=np.stack(points_translated)
    return points

def relative_robot_pos(robot_pos):
    '''
    robot_pos - [[x,y,theta]] scan_num x 3
    convert global robot pos to pos relative to last time
    the first is just 0,0,0
    '''
    rel_pos=np.zeros_like(robot_pos)
    for i in range(len(robot_pos)):
        if i>0:
            rel_pos[i]=robot_pos[i]-robot_pos[i-1]
    return rel_pos

def get_min_max_point(points,robot_pos):
    '''
    points - [[[x,y]]] scan_num x 181 x 2
    robot_pos - [[x,y,theta]] scan_num x 3
    a list of point lists relative to robots_pos
    return [[x_min,y_min],[x_max,y_max]]
    '''
    g_points=[robot_to_global(l,r) for (l,r) in zip(points,robot_pos)]
    g_points=np.stack(g_points) # scan_num x 181 x 2
    #print(g_points[:2])
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
    mmap=np.zeros((int(m_xrange),int(m_yrange)))
    mmap+=logoddsprior
    m_limit=np.zeros((2,2))
    m_limit[0,0]=0
    m_limit[0,1]=0
    m_limit[1,0]=mmap.shape[0]-1
    m_limit[1,1]=mmap.shape[1]-1
    return mmap,m_limit

def global_to_map(points,g_limit,m_limit):
    '''
    points - [[x,y]] 181 x 2 (181 can actually be any number)
    g_limit - [[x_min,y_min],[x_max,y_max]] for global coord
    m_limit - [[x_min,y_min],[x_max,y_max]] for map coord
    convert a global point to a pixel location on map
    assume both have the same aspect ratio
    '''
    x=points[:,0] # 181
    y=points[:,1] # 181
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
    m_points=np.stack([x_m,y_m],axis=1) # 181 x 2
    return m_points

def map_to_global(points,g_limit,m_limit):
    '''
    points - [[x,y]] 181 x 2 (181 can actually be any number)
    g_limit - [[x_min,y_min],[x_max,y_max]] for global coord
    m_limit - [[x_min,y_min],[x_max,y_max]] for map coord
    convert a map point to its global coord
    assume both have the same aspect ratio
    '''
    x=points[:,0] # 181
    y=points[:,1] # 181
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
    x_g=x_min+(x-mx_min)/m_xrange*g_xrange
    y_g=y_min+(y-my_min)/m_yrange*g_yrange
    g_points=np.stack([x_g,y_g],axis=1) # 181 x 2
    return g_points

def draw_map(mmap,fname="map.png"):
    '''
    output a png of drawn map for visualization
    '''
    w,h=mmap.shape
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,w*20,h*20)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.0, 0.0, 0.0)
    for i in range(w):
        for j in range(h):
            if mmap[i][j]>=logoddsthreshold:
                ctx.rectangle(i*20, j*20, 20, 20)
                ctx.fill()
    surface.write_to_png(fname)

def ray_tracing(mmap,robot_pos,g_limit,m_limit):
    '''
    given the current constructed map and robot pose
    computer the 181 points should be sensed by laser rangefinder
    mmap - w x h
    robot_pos - [x,y,theta]
    g_limit - [[x_min,y_min],[x_max,y_max]] for global coord
    m_limit - [[x_min,y_min],[x_max,y_max]] for map coord
    '''
    scan=np.zeros((181,2))
    pos_xy=np.expand_dims(robot_pos[:2],axis=0) # 1 x 2
    mpos_xy=global_to_map(pos_xy,g_limit,m_limit)[0] # 2
    theta=robot_pos[2] # current heading
    x,y=mpos_xy[0],mpos_xy[1]
    # scan from left to right for all 181 degrees
    for i in range(0,181):
        # scan from the location of the robot toward the edge of the map
        dis=1.0
        while True:
            heading=theta+(90-i)/180*pi
            loc_x=np.round(x+np.cos(heading)*dis)
            loc_y=np.round(y+np.sin(heading)*dis)
            if loc_x>=m_limit[1][0] or loc_y>=m_limit[1][1]:
                # out of range of the map, no occlusion found
                # set map boundary as occlusion
                # TODO: correctly handle out of range
                scan[i]=np.asarray([min(loc_x,m_limit[1][0]),min(loc_y,m_limit[1][1])])
                break
            elif mmap[int(loc_x)][int(loc_y)]>=logoddsthreshold:
                # occlusion found
                scan[i]=np.asarray([loc_x,loc_y])
                break
            dis+=1.0
    # convert scan to global coord
    scan_g=map_to_global(scan,g_limit,m_limit)
    return scan_g

if __name__ == '__main__':
    points,robpos=parse_file("jerodlab.2d")
    #print(points.shape)
    #print(robpos.shape)
    print("points:",points[0])
    print(robpos[:1])
    g_limit=get_min_max_point(points,robpos)
    #print(g_limit)
    mmap,m_limit=crate_map(g_limit,20)
    #print(mmap.shape)
    #print(m_limit)
    g_points=[robot_to_global(l,r) for (l,r) in zip(points,robpos)]
    g_points=np.stack(g_points)
    print("g_points:",g_points[0])
    m_points=[global_to_map(p,g_limit,m_limit) for p in g_points]
    m_points=np.stack(m_points)
    #print(m_points.shape)
    for i in range(len(m_points)):
        scan=m_points[i]
        for j in range(len(scan)):
            p=scan[j]
            mmap[int(p[0]),int(p[1])]=logoddsocc
        if i>10:
            break
    #print(mmap)
    #draw_map(mmap)
    test_rt=ray_tracing(mmap,np.asarray([0.0,0.0,3.1415926/2]),g_limit,m_limit)
    print("test:",test_rt)
