"""
Set here pathes, notebook configs and general utils

The project is organised with the below structure.
After seting up your own PATH_ROOT_DRIVE and PATH_ROOT_LOCAL
you can automaticallly setup the whole structure with initProject()

PATH_ROOT_DRIVE
-- Datasets (PATH_ROOT_DRIVE_DS)
   -- Images
      -> zipped image datasets
         dataset dowloadable from : https://doi.org/10.5281/zenodo.10465437 
   -- Events_Catalogs
      -> flare_catalog_file (PATH_FLCATALOG)
      -> flare_catalog_file_with_positions (PATH_FLCATALOG_WITH_POS)
      -> flare_with_unmatched_positions_dates (PATH_MISSING_POS_DATES)
         files dowloadable from : https://doi.org/10.5281/zenodo.10560188
   -- Time_Series
      -> windows_features_files (F_PATH_WINDOWS)
   -- Meta
      -> climatological_rates_file (PATH_CR_RATES)
-- Folds
   -- Flare
      -- Chunks (temporal_chunks_files F_PATH_CHUNKS)
      -- Allocation (maps_chunks_folds_alloc_files F_PATH_ALLOC)
      -- FoldsTrainVal (train_val_folds_files F_PATH_FOLDS)
      -- OperationalTest (test_set_files F_PATH_TEST)

PATH_ROOT_LOCAL
-- images (PATH_IMAGES)
   folder in which zipped files from 'PATH_ROOT_DRIVE_DS/Images' are unzipped


PATH_RES
-- Results_Paper_PCNN
   -> log.csv                  , models training status
   -> bests.csv                , best epoch-averages metrics
   -> bestsCVCrossEpoch.csv    , best cross-epoch metrics
   -> training_{modelName}.csv , fold-averages training and valiation curves
   -- training_folds
      -> training_{modelName}_fd{foldId}.csv , folds training and valiation curves
   -- models
      -- {modelName}            , saved trained model
      -> {modelName}_config.pkl , saved model config (metrics, optimizer, dataset parameters etc.)
   -- pred_test
      -> {modelName}_fd.csv            , full-disk predictions
      -> {modelName}_fd_withLabels.csv , full-disk predictions with labels
      -> {modelName}_pt.csv            , patches/sector predictions
      -> {modelName}_pf_withLabels.csv , patches/sector predictions with labels

"""

import sys
import os
from pathlib import Path

__all__ = [
    'PATH_SUNDL',
    'COLAB',
    'CLEAN_LOCAL',
    'MIXED_PREC',
    'PATH_ROOT_DRIVE',
    'PATH_ROOT_LOCAL',
    'PATH_ROOT_DRIVE_DS',
    'PATH_IMAGES',
    'PATH_FOLDS',
    'PATH_RES',
    'PATH_FLCATALOG',
    'PATH_CR_RATES',
    'PATH_MISSING_POS_DATES',
    'PATH_FLCATALOG_WITH_POS',
    'F_PATH_WINDOWS',
    'F_PATH_CHUNKS',
    'F_PATH_ALLOC',
    'F_PATH_FOLDS',
    'F_PATH_TEST',
    'F_PATH_PREDS',
    'F_PATH_LABEL_PATCHES',
    'initProject'
    ]

PATH_SUNDL = '../../'
PATH_PROJECT = f'{PATH_SUNDL}/notebooks/flare_limits_pcnn'
sys.path.append(PATH_SUNDL)
sys.path.append(PATH_PROJECT)

   
COLAB = False
CLEAN_LOCAL = False
MIXED_PREC = False

# ! keep the white spaces
PATH_ROOT_DRIVE = Path("/Users/greg/Google Drive/Mi unidad/Projects/Forecast")
PATH_ROOT_LOCAL = Path("/Users/greg/session")
  
if isinstance(PATH_ROOT_DRIVE,str):
  # ! do not put white spaces
  PATH_ROOT_DRIVE=Path(PATH_ROOT_DRIVE)
if isinstance(PATH_ROOT_LOCAL,str):
  PATH_ROOT_LOCAL=Path(PATH_ROOT_LOCAL)
  
PATH_ROOT_DRIVE_DS = PATH_ROOT_DRIVE/'Datasets'
PATH_IMAGES        = PATH_ROOT_LOCAL/'images'
PATH_FOLDS         = PATH_ROOT_DRIVE/'Folds/Flare/'
PATH_RES           = PATH_PROJECT #Path('./') #PATH_ROOT_DRIVE/'Results/Flare'

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

def initProject():
  if not PATH_ROOT_DRIVE.exists():
    raise Exception('Setup an existing PATH_ROOT_DRIVE in config.py')
  if not PATH_ROOT_LOCAL.exists():
    raise Exception('Setup an existing PATH_ROOT_LOCAL in config.py')
  
  if not PATH_IMAGES.exists():
    PATH_IMAGES.mkdir(parents=True, exist_ok=True)
  
  if not PATH_ROOT_DRIVE_DS.exists():
    PATH_ROOT_DRIVE_DS.mkdir(parents=True, exist_ok=True)
  for subFolder in ['Images','Events_Catalogs', 'Time_Series', 'Meta']:
    if not (PATH_ROOT_DRIVE_DS/subFolder).exists():
      os.mkdir(PATH_ROOT_DRIVE_DS/subFolder)
      
  if not PATH_FOLDS.exists():
    PATH_FOLDS.mkdir(parents=True, exist_ok=True)
  for subFolder in ['Chunks','Allocation', 'FoldsTrainVal', 'OperationalTest']:
    if not (PATH_FOLDS/subFolder).exists():
      os.mkdir(PATH_FOLDS/subFolder)
      
  
    