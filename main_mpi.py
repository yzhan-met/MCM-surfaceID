from lib.pixMaxBRF import pix_max_BRF
from lib.surfaceID import surface_id
import os
import configparser
import numpy as np

def make_maxBRF(DOY):
    config = configparser.ConfigParser()
    config.read_file(open('etc/config.txt'))
    maxBRF_filepath = config.get('general', 'SMB_folder')

    file_name = '{}/maxBRF_LA_{:03d}.nc'.format(maxBRF_filepath, DOY)
    if not os.path.exists(file_name):
        pix_max_BRF(DOY)

def make_surface_ID(maxBRF_filepath, maxBRF_file):

    try:
        surface_id("{}/{}".format(maxBRF_filepath, maxBRF_file))
        print("{} done".format(maxBRF_file))
    except Exception as e:
        print('not done; {} ; {}'.format(maxBRF_file,e))


if __name__=='__main__':
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #pixMaxBRF
    y = False#True
    if y:
        for r in range(size):
            if rank==r:
                DOY = np.arange(8,368,8)
                try:
                    make_maxBRF(DOY[r])
                    print('DOY {} done'.format(DOY[r]))
                except Exception as e:
                    print('DOY {} failed; error {}'.format(DOY[r], e))

    #surface_ID
    x = True
    if x:
        config = configparser.ConfigParser()
        config.read_file(open('etc/config.txt'))
        maxBRF_filepath = config.get('general', 'SMB_folder')
        maxBRF_file     = os.listdir(maxBRF_filepath)

        for r in range(size):
            if rank==r:
                DOY = np.arange(8,368,8)
                try:
                    make_surface_ID(maxBRF_filepath, maxBRF_file[r])
                    print('DOY {} done'.format(DOY[r]))
                except Exception as e:
                    print('DOY {} failed; error {}'.format(DOY[r], e))
