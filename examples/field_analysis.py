
import pickle
from voxel_helpers import *
from mayavi import mlab

data = pickle.load(open("/home/zenna/omdata/pdt/1490389836.2120135adt_scalar_field__/it_5000_fetch.pickle", "rb"))

input_voxels = data['extra_fetches']['voxel_grid.input_var']
output_voxels = data['extra_fetches']['decoded_vox_grid']

# voxel_grids = indices_voxels(input_voxels)
# dec_voxel_grids = indices_voxels(output_voxels)
#
# show_voxel_grid(dec_voxel_grids[2])
