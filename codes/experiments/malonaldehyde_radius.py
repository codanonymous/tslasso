import megaman
import numpy as np
from megaman.geometry import Geometry
from megaman.utils.estimate_radius import run_estimate_radius
import os
import datetime
import random
import sys
np.random.seed(0)
random.seed(0)
workingdirectory = os.popen('git rev-parse --show-toplevel').read()[:-1]
os.chdir(workingdirectory)
radius = 50
adjacency_method = 'cyflann'
adjacency_kwds = {'radius':radius}
affinity_method = 'gaussian'
affinity_kwds = {'radius':radius}
laplacian_method = 'symmetricnormalized'
laplacian_kwds = {'scaling_epps':radius}
dataname  = workingdirectory + '/untracked_data/chemistry_data/malonaldehydeangles022119_pca50'
#dataname = '/Users/samsonkoelle/Downloads/manigrad-100818/mani-samk-gradients/untracked_data/chemistry_data/ethanolangles022119_pca50'
data = np.load(dataname + '.npy')
geom = Geometry(adjacency_method=adjacency_method,adjacency_kwds=adjacency_kwds,affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
geom.set_data_matrix(data)
geom.laplacian_method = 'geometric'
geom.laplacian_kwds = {'scaling_epps':radius}
laplacian_matrix = geom.compute_laplacian_matrix()
geom.compute_laplacian_matrix()
sample = np.arange(0,data.shape[0],1000)
distorion_vs_rad_dim2 = run_estimate_radius(
data, geom.adjacency_matrix, sample=sample, d=2, rmin=1, rmax=10,
ntry=50, run_parallel=True, search_space='logspace')
np.save(dataname + '_distortionradius', distorion_vs_rad_dim2)