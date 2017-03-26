"""Helpers for Dealing with Voxels"""
import numpy as np
import os


def model_net_40(voxels_path=os.path.join(os.environ['DATADIR'],
                                          'ModelNet40',
                                          'alltrain32.npy')):
    voxel_grids = np.load(voxels_path)/255.0
    return voxel_grids


def voxel_indices(voxels, limit, missing_magic_num=-1):
    """
    Convert voxel data_set (n, 32, 32, 32) to (n, 3, m)
    """
    n, x, y, z = voxels.shape
    output = np.ones((n, 4, limit))
    output = output * missing_magic_num

    # obtain occupied voxels
    for v in range(len(voxels)):
        voxel = voxels[v]
        x_list, y_list, z_list = np.where(voxel)
        assert len(x_list) == len(y_list)
        assert len(y_list) == len(z_list)

        # fill in output tensor
        npoints = min(limit, len(x_list))
        output[v][0][0:npoints] = x_list[0:npoints]
        output[v][1][0:npoints] = y_list[0:npoints]
        output[v][2][0:npoints] = z_list[0:npoints]
        output[v][3][0:npoints] = voxel[x_list, y_list, z_list][0:npoints]

    output = np.transpose(output, [0, 2, 1]) # switch limit and coords

    return output


def indices_voxels(indices, grid_x=32, grid_y=32, grid_z=32):
    """Convert indices representation into voxel grid"""
    indices = indices.astype('int')
    nvoxels = indices.shape[0]
    # indices = np.transpose(indices, (0, 2, 1))
    voxel_grid = np.zeros((nvoxels, grid_x, grid_y, grid_z))
    for i in range(nvoxels):
        for j in range(indices.shape[1]):
            x = indices[i, j, 0]
            y = indices[i, j, 1]
            z = indices[i, j, 2]
            d = indices[i, j, 3]
            if 0 <= x < grid_x and 0 <= y < grid_y and 0 <= z < grid_z:
                voxel_grid[i, x, y, z] = d
            else:
                break

    return voxel_grid


def show_voxel_grid(grid):
    from mayavi import mlab
    # from mayavi import mlab
    """Vizualise voxel grid with mlab
    x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
    s = np.sin(x*y*z)/(x*y*z)
    """
    mlab.pipeline.volume(mlab.pipeline.scalar_field(grid), vmin=0, vmax=0.8)
    mlab.show()
