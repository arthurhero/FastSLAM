from copy import deepcopy
import numpy as np

def resampling(particles):
    '''
    particles - a list of particles
    return a list of resampled particles
    '''
    weights=np.asarray([p.weight for p in particles]) # all the weights
    if -1 in weights:
        # if we manually told the algorithm not to resample
        return particles
    #poses=np.asarray([p.pos for p in particles]) # all the poses 
    #print("weights:",weights)
    #print("poses:",poses)
    # TODO: do not resample if weight variance is small
    normalized_weights=weights/weights.sum()
    resampled_freq=np.random.multinomial(len(particles),normalized_weights) # freq being resampled for each particle
    resampled_particles=list()
    for i in range(len(resampled_freq)):
        f=resampled_freq[i]
        for j in range(f):
            resampled_particles.append(deepcopy(particles[i]))
    return resampled_particles
