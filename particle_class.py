import math
import numpy as np
from utility import *
from scan_match import icp,search_best_angle

class particle:
    '''
    a particle class
    each particle maintain its own map, robot position and its own weight
    '''
    # initialize robot position at the origin and face the front (90 degrees)
    pos = np.zeros(3,)
    pos[2]=pi/2.0

    def __init__(self,num_particles,mmap,g_limit,m_limit,motion_noise=2,theta_noise=1,laser_noise=1.5):
        '''
        num_particles - total number of particles existing
        mmap - current map for this particle
        g_limit - [[x_min,y_min],[x_max,y_max]] for global coord
        m_limit - [[x_min,y_min],[x_max,y_max]] for map coord
        motion_noise - sigma for motion in mm
        theta_noise - sigma for theta in degree
        laser_noise - sigma for laser in mm (currently not used)
        '''
        self.weight = 1.0/num_particles	
        self.grid_map = mmap
        self.g_limit=g_limit
        self.m_limit=m_limit
        self.motion_noise=motion_noise
        self.theta_noise=theta_noise
        self.laser_noise=laser_noise

    def update_pose(self,rel_pos,robscan,valid_scan,scan_match=False):
        '''
        update the robot's position according to odometry feedback and scan matching
        rel_pos - 3, motion relative to self.pos, odometry feedback from the robot
        robscan - 181 x 2, actual sensor data relative to robot
        valid_scan - indices of valid scan
        scan_match - whether to perform scan_match or not
        '''
        # sample noises
        x_noise=np.random.normal(loc=0.0,scale=self.motion_noise)
        y_noise=np.random.normal(loc=0.0,scale=self.motion_noise)
        t_noise=np.random.normal(loc=0.0,scale=self.theta_noise)
        t_noise_rad=t_noise/180*pi
        noises=[x_noise,y_noise,t_noise_rad]
        # update position
        new_pos=self.pos+rel_pos+noises
        if not scan_match:
            self.pos=new_pos
            return
        # perform scan-matching
        scan=robot_to_global(robscan,new_pos)
        angle=search_best_angle(scan,valid_scan,self.grid_map,new_pos,self.g_limit,self.m_limit,-1,2,1)
        #print("angle:",angle)
        theta=angle*pi/180
        new_pos[2]+=(theta*1.0)
        self.pos=new_pos
        return

    def update_weight(self,robscan,valid_scan):
        '''
        get weight according to the raytracing result
        the smaller the mismatch, the higher the weight score
        robscan - 181 x 2, actual sensor data relative to robot
        valid_scan - indices of valid scan
        '''
        scan=robot_to_global(robscan,self.pos)
        ps,valid_idx=ray_tracing(self.grid_map,self.pos,self.g_limit,self.m_limit) # 181 x 2, supposed sensor data
        #print(valid_idx)
        valid_idx=set(valid_idx).intersection(set(valid_scan))
        valid_idx=list(valid_idx)
        if len(valid_idx)<50:
            self.weight=-1 # invalid scan due to sparse map, does not resample this round
            return
        ps=ps[valid_idx]
        scan=scan[valid_idx]
        error=np.mean(np.power(np.sum((scan-ps)**2,axis=1),0.5))
        weight = 1.0/(error + 1)
        self.weight=weight
        return 

    def update_map(self,robscan,valid_scan):
        '''
        update map according to the raytracing result
        if the grid occupied, increase of occlusion probability
        if the ray get through a grid, decrease the occlusion probability
        robscan - 181 x 2, actual sensor data relative to robot
        valid_scan - indices of valid scan
        '''
        # convert everything to map coord
        pos_xy=np.expand_dims(self.pos[:2],axis=0) # 1 x 2
        mpos_xy=global_to_map(pos_xy,self.g_limit,self.m_limit)[0] # 2
        scan=robot_to_global(robscan,self.pos)
        map_scan=global_to_map(scan,self.g_limit,self.m_limit) # 181 x 2
        for vs in valid_scan:
            ms=map_scan[vs]
            # get the angle
            if (ms[0]-mpos_xy[0])==0.0:
                # handle the cases invalid for arctan
                if (ms[1]-mpos_xy[1])==0.0:
                    continue # do not update map for this scan
                elif (ms[1]-mpos_xy[1])>0.0:
                    theta=pi/2.0
                else:
                    theta=-pi/2.0
            else:
                theta=np.arctan((ms[1]-mpos_xy[1])/(ms[0]-mpos_xy[0]))
                if (ms[0]-mpos_xy[0])<0:
                    theta=pi+theta
            #print("theta:",theta)
            # calculate distance
            dis=np.power(np.sum((ms-mpos_xy)**2),0.5)
            cur_dis=0.0
            while True:
                cur_x=mpos_xy[0]+np.round(np.cos(theta)*cur_dis)
                cur_y=mpos_xy[1]+np.round(np.sin(theta)*cur_dis)
                if cur_x >= self.m_limit[0][0] and cur_x <= self.m_limit[1][0]\
                        and cur_y >= self.m_limit[0][1] and cur_y <= self.m_limit[1][1]:
                    #inside the boundary
                    if self.grid_map[int(cur_x)][int(cur_y)]>=logoddsupdate:
                        #if detected occlusion
                        break # do not update (trust past scans)
                    if cur_dis<dis-0.5:
                        self.grid_map[int(cur_x),int(cur_y)]+=logoddsfree
                    else:
                        self.grid_map[int(cur_x),int(cur_y)]+=logoddsocc
                        break
                    cur_dis+=1.0
                else:
                    break
        return
