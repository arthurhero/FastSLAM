from copy import deepcopy
import numpy as np
from utility import *
from particle_class import particle
from resampling import resampling

if __name__ == '__main__':
    # test code for any .2d file
    filename="jerodlab.2d"
    num_particles = 30
    progress_folder = "progress_sm_1_1_no_del/"
    result_folder = "sm1_1_no_del/"
    use_scan_match=True
    map_reso = 50 # resolution of the grip map (mm)

    # prepare the data (robot position and scan history)
    robscan,robpos=parse_file(filename)
    g_limit=get_min_max_point(robscan,robpos)
    mmap,m_limit=create_map(g_limit,map_reso)
    rel_robpos=relative_robot_pos(robpos)

    # construct the particles
    particles = [particle(num_particles,deepcopy(mmap),g_limit,m_limit) for i in range(num_particles)]

    #  start update
    for i in range(len(robscan)):
        print(i)
        for j in range(num_particles):
            if i>0:
                # only update pose and weight after first round
                particles[j].update_pose(rel_robpos[i],robscan[i],scan_match=True)
                particles[j].update_weight(robscan[i])
            particles[j].update_map(robscan[i])
            '''
                '''
            # draw some map in progress
            if i>0 and j<4:
                draw_map(particles[j].grid_map,progress_folder+str(i)+"_"+str(j)+".png")
        '''
            '''
        if i>0:
            # resample the particles after all weights are updated
            particles = resampling(particles)

    for i in range(num_particles):
        draw_map(particles[i].grid_map,result_folder+str(i)+".png")
