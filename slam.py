import os
from os import path

from copy import deepcopy
import numpy as np
from utility import *
from particle_class import particle
from resampling import resampling

if __name__ == '__main__':
    # test code for any .2d file
    filename="jerodlab.2d"
    num_particles = 30
    result_folder = "tmp"
    progress_folder = result_folder+'_progress'
    use_scan_match=True
    map_reso = 40 # resolution of the grip map (mm)

    if not path.isdir(result_folder):
        os.mkdir(result_folder)
    if not path.isdir(progress_folder):
        os.mkdir(progress_folder)

    result_folder+='/'
    progress_folder+='/'

    # prepare the data (robot position and scan history)
    robscan,robpos,valid_scan=parse_file(filename)
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
                particles[j].update_pose(rel_robpos[i],robscan[i],valid_scan[i],scan_match=use_scan_match)
                particles[j].update_weight(robscan[i],valid_scan[i])
            particles[j].update_map(robscan[i],valid_scan[i])
            '''
                '''
            # draw some map in progress
            if i>0 and j<4:
                draw_map(particles[j].grid_map,progress_folder+str(i)+"_"+str(j)+".png",greyscale=True)
        '''
            '''
        if i>0:
            # resample the particles after all weights are updated
            particles = resampling(particles)

    for i in range(num_particles):
        draw_map(particles[i].grid_map,result_folder+'/'+str(i)+".png",greyscale=False)
