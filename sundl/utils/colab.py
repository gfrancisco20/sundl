"""
Utilities and constants for Google Colab
"""

import os
from pathlib import Path
import shutil
import time

__all__ = ['ressourcesSetAndCheck',
           'mountDrive',
           'drive2local'
           ]

def drive2local(files2transfer
                ):
  """
  Parameters
  ----------
  files2transfer : dict
    files2transfer['typeFiles'] = (sourceDir, destDir, listOfFilenames)
  """
  for typeFiles in files2transfer.keys():
    sourceDir, destDir, listOfFilename = files2transfer[typeFiles]
    if isinstance(sourceDir,str): sourceDir = Path(sourceDir)
    if isinstance(destDir,str): destDir = Path(destDir)
    if not destDir.exists():
      # os.mkdir(destDir.as_posix())
      os.makedirs(destDir, exist_ok=True)
      duration = time.time()
      for fn in listOfFilename:
        zipfile = False
        if len(fn.split('.'))==1:
          # if fn giveen without extention it is assumed to be a zip file
          fn = f'{fn}.zip'
          zipfile = True
        elif fn[-3:] == 'zip':
          zipfile = True
        destPath = destDir/fn.split('/')[-1]
        shutil.copy((sourceDir/fn).as_posix(), destPath.as_posix())
        duration = time.time()-duration
        print(f'\n{typeFiles} -- {fn} transfered in : {duration//60:.0f} m {duration%60:.2f} s')

        if zipfile:
          duration = time.time()

          shutil.unpack_archive(destPath.as_posix(), destDir, "zip")
          os.remove(destPath.as_posix())

          duration = time.time()-duration
          print(f'    {fn} unzipped in : {duration//60:.0f} m {duration%60:.2f} s')
    else:
      print(f'\n{typeFiles} files already in local')

def mountDrive():
  if not Path('/content/drive').exists():
    from google.colab import drive
    drive.mount('/content/drive')

def ressourcesSetAndCheck(mixedPrec = False):
  print('')
  gpu_info = os.system('nvidia-smi') #!nvidia-smi
  if gpu_info == 32512:
    print('Not connected to a GPU')
  else:
    gpu_info = '\n'.join(gpu_info)
    if gpu_info.find('failed') >= 0:
      print('Not connected to a GPU')
    else:
      print(gpu_info)
      print('List of GPUs running in Tensorflow :')
      import tensorflow as tf
      print(tf.config.list_physical_devices('GPU'))
      if mixedPrec:
        # Mixed precision policy (only usefull if tpu or gpu with compute capability of at least 7.0)
        # ex: not usefull for K80, some P100 (after some test aparantly all of them here...)
        print('Turning on mixed precision policy')
        policy = tf.keras.mixed_precision.Policy('mixed_float16') # TODO handle TPU case with mixed_bfloat16
        tf.keras.mixed_precision.set_global_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)
        print('WARNING : Do not forget to set output layers in float32 (better numerical stability)')
        print('    --->  outputs = layers.Activation(\'softmax\', dtype=\'float32\', name=\'predictions\')(x)')
        print('WARNING : For custom training loop, use tf.keras.mixed_precision.LossScaleOptimizer to use an appropriate loss scaling strategy')
        print('          for more information see - https://www.tensorflow.org/guide/mixed_precision')
        print('WARNING : For better performances, the following object parameters should be multiple of 8')
        print('          (Dense/LSTM , units) - (Conv2d , filters) - (Model.fit , batch_size')
  from psutil import virtual_memory
  ram_gb = virtual_memory().total / 1e9
  print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

  if ram_gb < 20:
    print('Not using a high-RAM runtime')
  else:
    print('You are using a high-RAM runtime!')
