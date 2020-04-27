
from lib.pixMaxBRF import pix_max_BRF
from lib.surfaceID import surface_id
import os
import configparser
# pix_max_BRF(8)

#make surfce ID for all available maxBRF_xxx.nc files for each DOY bin
config = configparser.ConfigParser()
config.read_file(open('etc/config.txt'))
maxBRF_filepath = config.get('general', 'SMB_folder')
maxBRF_file_avail = os.listdir(maxBRF_filepath)
#print(maxBRF_file_avail)
for maxBRF_file in maxBRF_file_avail[1:]: 
    try:
        surface_id("{}/{}".format(maxBRF_filepath, maxBRF_file))
#maxBRF_file = maxBRF_file_avail[2]    
#surface_id("{}/{}".format(maxBRF_filepath, maxBRF_file))
        print("{} done".format(maxBRF_file))
    except Exception as e:
        print('not done; {} ; {}'.format(maxBRF_file,e))
