import numpy as np
from utility import *
from icp import icp
from resampling import resampling

class particle:
    pos = np.zeros(3,)

    def __init__(self,num_particles,mmap,g_limit,m_limit,motion_noise=3,theta_noise=3,laser_noise=1.5):
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
        ps=ray_tracing(self.grid_map,new_pos,self.g_limit,self.m_limit) # 181 x 2, supposed sensor data
        scan=robot_to_global(robscan,new_pos)
        R,t,E=icp(scan,ps)
        if E>1.0:
            self.pos=new_pos
            return
        xy_pos=np.expand_dims(new_pos[:2],axis=1) # 2 x 1
        matched_pos=t+np.matmul(R,xy_pos) # 2 x 1
        matched_pos_=np.zeros(3,)
        matched_pos_[:2]=matched_pos[:,0]
        matched_pos_[2]=new_pos[2]
        self.pos=matched_pos_
        return

    def update_weight(self,robscan):
        '''
        get weight according to the raytracing result
        robscan - 181 x 2, actual sensor data relative to robot
        '''
        scan=robot_to_global(robscan,self.pos)
        ps=ray_tracing(self.grid_map,self.pos,self.g_limit,self.m_limit) # 181 x 2, supposed sensor data
        error=np.power(np.sum((scan-ps)**2),0.5)
        weight = 1.0/(error + 1)
        self.weight=weight
        return 

    def update_map(self,robscan):
        '''
        update map according to the raytracing result
        scan - 181 x 2, actual sensor data relative to robot
        '''
        update_map=np.zeros(np.shape(self.grid_map))
        update_map+=logoddsprior
        pos_xy=np.expand_dims(self.pos[:2],axis=0) # 1 x 2
        mpos_xy=global_to_map(pos_xy,self.g_limit,self.m_limit)[0] # 2
        scan=robot_to_global(robscan,self.pos)
        map_scan=global_to_map(scan,self.g_limit,self.m_limit) # 181 x 2
        for ms in map_scan:
            theta=np.arctan((ms[1]-mpos_xy[1])/(ms[0]-mpos_xy[0]))
            dis=np.power(np.sum((ms-mpos_xy)**2),0.5)
            cur_dis=0.0
            while cur_dis<dis:
                cur_x=mpos_xy[0]+np.round(np.cos(theta)*cur_dis)
                cur_y=mpos_xy[1]+np.round(np.sin(theta)*cur_dis)
                if cur_x >= self.m_limit[0][0] and cur_x <= self.m_limit[1][0]\
                        and cur_y >= self.m_limit[0][1] and cur_y <= self.m_limit[1][1]:
                    update_map[int(cur_x),int(cur_y)]=logoddsfree
                    cur_dis+=1.0
                else:
                    break
            if ms[0]>=self.m_limit[0][0] and ms[0]<=self.m_limit[1][0]\
                    and ms[1]>=self.m_limit[0][1] and ms[1]<=self.m_limit[1][1]:
                update_map[int(ms[0]),int(ms[1])]=logoddsocc
        self.grid_map-=logoddsprior
        self.grid_map+=update_map
        return

if __name__ == '__main__':
    num_particles = 30
    robscan,robpos=parse_file("jerodlab.2d")
    g_limit=get_min_max_point(robscan,robpos)
    mmap,m_limit=crate_map(g_limit,20)
    rel_robpos=relative_robot_pos(robpos)

    particles = [particle(num_particles,mmap,g_limit,m_limit) for i in range(num_particles)]

    for i in range(len(robscan)):
        for j in range(num_particles):
            if i>0:
                particles[j].update_pose(rel_robpos[i],robscan[i],scan_match=False)
                particles[j].update_weight(robscan[i])
            particles[j].update_map(robscan[i])
        if i>0:
            particles = resampling(particles)

    draw_map(particles[0].grid_map,"naive_fastslam.png")
