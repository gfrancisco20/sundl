"""
Set here pathes, notebook configs and utils
"""

import sys
import os
from pathlib import Path

PATH_SUNDL = '../../'
sys.path.append(PATH_SUNDL)

   
COLAB = False
CLEAN_LOCAL = False
MIXED_PREC = False

if COLAB:
  PATH_ROOT_DRIVE = Path("/content/drive/MyDrive/Projects/Forecast")
  PATH_ROOT_LOCAL = Path("/content/session")
else:
  PATH_ROOT_DRIVE = Path("/Users/greg/Google Drive/Mi unidad/Projects/Forecast")
  PATH_ROOT_LOCAL = Path("/Users/greg/session")
  if not PATH_ROOT_LOCAL.exists():
    os.makedirs(PATH_ROOT_LOCAL)
  
PATH_ROOT_DRIVE_DS = PATH_ROOT_DRIVE/'Datasets'
PATH_IMAGES        = PATH_ROOT_LOCAL/'images'
PATH_FOLDS         = PATH_ROOT_DRIVE/'Folds/Flare/'
PATH_RES           = Path('./') #PATH_ROOT_DRIVE/'Results/Flare'

PATH_FLCATALOG = PATH_ROOT_DRIVE_DS/f'Events_Catalogs/flare_catalog_2023_04__1986_01.csv'
PATH_CR_RATES  = PATH_ROOT_DRIVE_DS/'Meta/fl_climatology.pkl'
PATH_MISSING_POS_DATES  = PATH_ROOT_DRIVE_DS/f'Events_Catalogs/missing_date_pos_CMX_2020.pkl'
PATH_FLCATALOG_WITH_POS = PATH_ROOT_DRIVE_DS/f'Events_Catalogs/flare_positions_2020_C.csv'
 
F_PATH_WINDOWS= lambda labelCol, window_h: PATH_ROOT_DRIVE_DS/f'Time_Series/windowsHistory_{labelCol}_{window_h}h.csv'
F_PATH_CHUNKS = lambda labelCol, window_h: PATH_FOLDS/f'Chunks/chunks_{labelCol}_{window_h}h.csv'
F_PATH_ALLOC  = lambda labelCol, window_h: PATH_FOLDS/f'Allocation/foldsChunkAllocation_{labelCol}_{window_h}h.csv'
F_PATH_FOLDS  = lambda labelCol, window_h: PATH_FOLDS/f'FoldsTrainVal/dfFoldsTrainVal_{labelCol}_{window_h}h.csv'
F_PATH_TEST   = lambda labelCol, window_h: PATH_FOLDS/f'OperationalTest/dfTest_{labelCol}_{window_h}h.csv'

F_PATH_PREDS = lambda resDir : resDir/'pred_test'
def F_PATH_PREDS_MODEL(predDir, modelName, predType = 'fd', withLabels = False):
  labelTag = ''
  if withLabels:
    labelTag = '_withLabels'    
  return predDir/f'{modelName}_{predType}{labelTag}.csv'

F_PATH_LABEL_PATCHES = lambda num_patches, timeRes_h, window_h:  PATH_ROOT_DRIVE_DS/f'Labels/patches_x{num_patches}_history_{timeRes_h}x{window_h}_2020_t2f.pkl'
