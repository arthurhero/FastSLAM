import math
from copy import deepcopy
import numpy as np
from utility import *
from scan_match import icp,search_best_angle
from resampling import resampling

class particle:
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
        laser_noise - sigma for laser in mm
        '''
        self.weight = 1.0/num_particles	
        self.grid_map = mmap
        self.g_limit=g_limit
        self.m_limit=m_limit
        self.motion_noise=motion_noise
        self.theta_noise=theta_noise
        self.laser_noise=laser_noise

    def update_pose(self,rel_pos,robscan,scan_match=False):
        '''
        rel_pos - 3, motion relative to self.pos
        robscan - 181 x 2, actual sensor data relative to robot
        scan_match - whether to perform scan_match or not
        '''
        x_noise=np.random.normal(loc=0.0,scale=self.motion_noise)
        y_noise=np.random.normal(loc=0.0,scale=self.motion_noise)
        t_noise=np.random.normal(loc=0.0,scale=self.theta_noise)
        t_noise_rad=t_noise/180*pi
        noises=[x_noise,y_noise,t_noise_rad]
        new_pos=self.pos+rel_pos+noises
        if not scan_match:
            self.pos=new_pos
            return
        # perform scan-matching
        scan=robot_to_global(robscan,new_pos)
        angle=search_best_angle(scan,self.grid_map,new_pos,self.g_limit,self.m_limit,-1,2,1)
        #print("angle:",angle)
        theta=angle*pi/180
        '''
        ps,valid_idx=ray_tracing(self.grid_map,new_pos,self.g_limit,self.m_limit) # 181 x 2, supposed sensor data
        if len(valid_idx)<170:
            self.pos=new_pos
            return
        ps=ps[valid_idx]
        scan=scan[valid_idx]
        #theta,t,E=icp(scan,ps,new_pos[:2])
        theta,t,E=icp(ps,scan,new_pos[:2])
        xy_pos=np.expand_dims(new_pos[:2],axis=1) # 2 x 1
        matched_pos=xy_pos+t # 2 x 1
        matched_pos_=np.zeros(3,)
        matched_pos_[:2]=matched_pos[:,0]
        matched_pos_[2]=new_pos[2]+theta
        middle_pos=(new_pos+matched_pos_)/2.0
        #self.pos=middle_pos
        self.pos=matched_pos_
        '''
        new_pos[2]+=(theta*1.0)
        self.pos=new_pos
        return

    def update_weight(self,robscan):
        '''
        get weight according to the raytracing result
        robscan - 181 x 2, actual sensor data relative to robot
        '''
        scan=robot_to_global(robscan,self.pos)
        ps,valid_idx=ray_tracing(self.grid_map,self.pos,self.g_limit,self.m_limit) # 181 x 2, supposed sensor data
        #print(valid_idx)
        if len(valid_idx)<50:
            self.weight=-1 # invalid error, does not resample this round
            return
        ps=ps[valid_idx]
        scan=scan[valid_idx]
        error=np.mean(np.power(np.sum((scan-ps)**2,axis=1),0.5))
        weight = 1.0/(error + 1)
        self.weight=weight
        return 

    def update_map(self,robscan):
        '''
        update map according to the raytracing result
        scan - 181 x 2, actual sensor data relative to robot
        '''
        pos_xy=np.expand_dims(self.pos[:2],axis=0) # 1 x 2
        mpos_xy=global_to_map(pos_xy,self.g_limit,self.m_limit)[0] # 2
        scan=robot_to_global(robscan,self.pos)
        map_scan=global_to_map(scan,self.g_limit,self.m_limit) # 181 x 2
        for ms in map_scan:
            if (ms[0]-mpos_xy[0])==0.0:
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
            dis=np.power(np.sum((ms-mpos_xy)**2),0.5)
            cur_dis=0.0
            while True:
                cur_x=mpos_xy[0]+np.round(np.cos(theta)*cur_dis)
                cur_y=mpos_xy[1]+np.round(np.sin(theta)*cur_dis)
                if cur_x >= self.m_limit[0][0] and cur_x <= self.m_limit[1][0]\
                        and cur_y >= self.m_limit[0][1] and cur_y <= self.m_limit[1][1]:
                    #inside the boundary
                    if self.grid_map[int(cur_x)][int(cur_y)]>=logoddsthreshold:
                        #if detected occlusion
                        break # do not update
                    if cur_dis<dis-0.5:
                        self.grid_map[int(cur_x),int(cur_y)]+=logoddsfree
                    elif cur_dis>dis+0.5:
                        #unknown beyond the occlusion
                        self.grid_map[int(cur_x),int(cur_y)]*=0.8
                    else:
                        self.grid_map[int(cur_x),int(cur_y)]+=logoddsocc
                        break
                        '''
                        if self.grid_map[int(cur_x)][int(cur_y)]>(-logoddsthreshold):
                            #if not detected free space
                        '''
                    cur_dis+=1.0
                else:
                    break
        return

if __name__ == '__main__':
    num_particles = 30
    robscan,robpos=parse_file("jerodlab.2d")
    g_limit=get_min_max_point(robscan,robpos)
    mmap,m_limit=create_map(g_limit,map_reso)
    rel_robpos=relative_robot_pos(robpos)

    particles = [particle(num_particles,deepcopy(mmap),g_limit,m_limit) for i in range(num_particles)]

    for i in range(len(robscan)):
        print(i)
        #print(i,":",(particles[0].pos[2]-pi/2)/pi*180)
        for j in range(num_particles):
            if i>0:
                particles[j].update_pose(rel_robpos[i],robscan[i],scan_match=True)
                particles[j].update_weight(robscan[i])
            particles[j].update_map(robscan[i])
            '''
                '''
            if i>0 and j<4:
                draw_map(particles[j].grid_map,"progress_sm_1_1_no_del/"+str(i)+"_"+str(j)+".png")
        '''
            '''
        if i>0:
            particles = resampling(particles)

    for i in range(num_particles):
        draw_map(particles[i].grid_map,"sm1_1_no_del/"+str(i)+".png")
