"""
"""


from glob import glob
import pathlib
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sundl.utils.data import read_Dataframe_With_Dates
def configure_for_performance(ds, batch_size, shuffle_buffer_size=1000, shuffle = False, cache=True, prefetch=True, epochs = None, cachePath = ''):
  if cache:
    ds = ds.cache(cachePath)
  if shuffle:
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  ds = ds.batch(batch_size)
  if prefetch:
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds
  
def fileId2FnPattern(fileId,pathDir,channels):
  for idx,chan in enumerate(channels):
    if type(pathDir) == pathlib.PosixPath: pathDir = pathDir.as_posix()
    regexp = tf.strings.join([pathDir,'/*/',fileId,'_',str(chan),'.*'])#.numpy()
    # pth = sorted(glob(regexp))[0] # --> not supported by tf map
    if idx==0:
      #res = regexp
      res = tf.expand_dims(regexp,axis=-1)
    #elif idx==1:
      #print(res.shape)
      #print(regexp.shape)
    #  res = tf.stack([res,regexp],axis=-1)
    else:
      #print(res.shape)
      #print(regexp.shape)
      res = tf.concat([res,[regexp]],axis=-1)
  return res

def parse_image(file_path,pathDir, gray2RGB, isGray = False, sepPosNeg=False):
  # Load the raw data from the file as a string
  for idx in range(file_path.shape[0]):
    img = tf.io.read_file(file_path[idx])
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=0)
    # !! Resize automatically converts to float32 while preserving original values, while cast normalize values between [0-1]
    # if img_size is not None:
    #   img = tf.image.resize(img, [img_size, img_size])
    # else:
    img = 255.0*tf.image.convert_image_dtype(img, tf.dtypes.float32, saturate=False, name=None)
    if idx==0:
      res = img
    else:
      res = tf.concat([res,img],axis=-1)
  if isGray:
    if sepPosNeg:
      pos = tf.math.maximum(res,127)
      neg = tf.math.minimum(res,127)
      res =  tf.concat([neg,res,pos],axis=-1)
    elif gray2RGB:
      res = tf.repeat(res,repeats=3,axis=-1)
  return res

#@tf.autograph.experimental
def parse_image_normalize_tfMode(file_path, gray2RGB, means, stds):
  for idx in range(file_path.shape[0]):
    img = tf.io.read_file(file_path[idx])
    img = tf.io.decode_jpeg(img, channels=0)
    # if img_size is not None:
    #   img = tf.image.resize(img, [img_size, img_size])
    # else:
    img = 255.0*tf.image.convert_image_dtype(img, tf.dtypes.float32, saturate=False, name=None)
    # chnWiseNormalize:
    img = (img - means[idx]) / stds[idx]
    if idx==0:
      res = img
    else:
      res = tf.concat([res,img],axis=-1)
  if  res.shape[-1]==1:
    if gray2RGB :
      res = tf.repeat(res,repeats=3,axis=-1)
  return res

def builDS_image_feature(
  pathDir,
  channels,
  batch_size,
  dfTimeseries,
  samples, # sample dates
  shiftSamplesByLabelOff = False,
  ts_off_label_hours= 24*np.arange(0.5,6.5,0.5),
  ts_off_scalar_hours= None, 
  labelCol          = 'sw_v',
  scalarCol         = None,
  prefetch          = True,
  cache             = True,
  cachePath         = '', # keep string empty for ram caching
  shuffle           = True,
  uncachedShuffBuff = 1000,
  img_size          = None,
  crop_coord        = None,
  num_classes       = 2,
  gray2RGB          = True,
  sepPosNeg         = False,
  shape3d           = False,
  regression        = False,
  labelEncoder      = None,
  encoderIsTf       = True,
  scalarEncoder     = None,
  weightByClass     = False,
  weightOffLabIdx   = 0,
  # needed only if weightByClass is True, class are defined relatively to labelCol
  classTresholds    = {'quiet': (0,1e-7), 
                       'B':(1e-7,1e-6), 
                       'C':(1e-6,1e-5), 
                       'M':(1e-5,1e-4), 
                       'X':(1e-4,np.inf)},
  classWeights      = {'quiet': 0.20, 
                       'B':0.20, 
                       'C':0.20, 
                       'M':0.20, 
                       'X':0.20},
  strictly_pos_label = True,
  dates2exclude = None,
  **kwargs # bacckward compt
):
  if labelEncoder is not None and encoderIsTf:
    temp = labelEncoder
    labelEncoder = lambda x: temp(x).numpy()
  dfTimeseries = dfTimeseries.copy()
  if samples is not None:
    samples = samples.copy()
  if scalarCol is None:
    scalarCol = labelCol
  if type(ts_off_label_hours) not in [np.ndarray,list]:
    ts_off_label_hours = [ts_off_label_hours]
  if ts_off_scalar_hours is not None:
    if type(ts_off_scalar_hours) not in [np.ndarray,list]:
      ts_off_scalar_hours = [ts_off_scalar_hours]
  '''
  Generate a tensorflow dataset of images and features
  Parameters
  ----------
  channels : `list` of `string` or `int`
    images channel ID to read and assemble from directory 'pathDir'
    
  dfTimeseries : DataFrame
    continuous timeseries containing labels and eventual features,
    features can be added in this dataset only from dfTimeseries
    
  samples : DataFrame
    DataFrame where the index contain the date of the samples to be used in the dataset
    
  shiftTsByLabOff : bool, optional
    shift the 'dfTimeseries' dates of 'ts_off_label_hours' hours, 
    needed if dfTimeseries is given at feature levels, 
    i.e. values at date D gives feature of timestep 0 for date D and not labels,
    in the case of flare forecasting the features of timestep 0 
    characterize the time-window [D-windowSize ; D[,
    while the labels values refer to [D ; D + windowSize[,
    default to true
    
  shiftSamplesByLabelOff : bool, optional
    shift the 'sample' dates of 'ts_off_label_hours' hours, 
    needed if the sample dates where computed at feature level rather than featur,
    default to true

  gray2RGB : `bool`, optional
    only for one channel images (i.e. len(channel)==len(timesteps)==1)
    convert the resulting 1 channel image into a 3 channel one by repeating it
    usefull to use premade NN on 1 channel images

  Returns
  -------
  `tf.data.Dataset`
    a tensorflow dataset where an image is of the shape : [img_size,img_size,len(channels)+len(timesteps)]
    the channels or organized as follow :
    channel_1
    ...
    channel_n 
  '''
  # 
  if len(channels)>2: 
    gray2RGB=False
  if isinstance(pathDir,pathlib.PosixPath): 
    pathDir = pathDir.as_posix()


    
  # offseting
  # if shiftTsByLabOff:
  for offLabel in ts_off_label_hours:
    dfTimeseries[f'label_{offLabel}'] = dfTimeseries[labelCol].rolling(
            window = f'{offLabel}H',
            closed = 'right', # min_periods = int(scalar_lag)
            ).apply(lambda x: x[-1]).shift(freq = f'-{offLabel}H')#[:-int(offLabel/2)] 
    numna = dfTimeseries[f'label_{offLabel}'].isna().sum()
    print(f'WARNING : {numna} NaN (droped) for label at ts {offLabel}')
    
  if ts_off_scalar_hours is not None:
    for offScalar in ts_off_scalar_hours:
      # WARNING : offScalar is expeccted negative or null
      scalar_lag = -offScalar #- int(offScalar//24)
      dfTimeseries[f'scalar_{offScalar}'] = dfTimeseries[scalarCol].rolling(
          window = f'{scalar_lag}H',
          closed = 'both', # min_periods = int(scalar_lag)
          ).apply(lambda x: x[0])#[24*scalar_lag:] 

  if ts_off_scalar_hours is not None:
    startIdx = int(np.max(np.abs(ts_off_scalar_hours))/2)
  else:
    startIdx = 0
    
  dfTimeseries = dfTimeseries[startIdx : -int(np.max(ts_off_label_hours)/2)]  
  dfTimeseries.dropna(subset=[f'label_{offLabel}' for offLabel in ts_off_label_hours])
      
  # fiiltering on sample dates
  if samples is not None:
    if shiftSamplesByLabelOff:
      print('Samples shiiftng done')
      # use when the balance of the sample is made on the actual window values, not their foreccast-labels
      samples.index = samples.index + pd.DateOffset(hours= -ts_off_label_hours[0])
    dfTimeseries = dfTimeseries[dfTimeseries.index.isin(samples.index)]
    
  if dates2exclude is not None:
    dfTimeseries = dfTimeseries[~dfTimeseries.index.isin(dates2exclude)]

  if not cache and shuffle:
    # necessary shuffle sumplement with small buffer
    dfTimeseries = dfTimeseries.sample(frac=1)
    shuffle_buffer_size = len(dfTimeseries)
  else:
    shuffle_buffer_size = uncachedShuffBuff

  dfTimeseries = dfTimeseries.reset_index()
  dfTimeseries['id'] = dfTimeseries['timestamp'].apply(lambda x: x.strftime('%Y%m%d_%H%M'))
  dfTimeseries['pth'] = dfTimeseries['timestamp'].apply(lambda x: x.strftime('%Y/%m/%d'))
  dfTimeseries = dfTimeseries.set_index('timestamp')
  fullPthIds = dfTimeseries[['pth','id']].apply(lambda x: x['pth']+'/'+x['id'],axis=1)
  fileIds_ds = tf.data.Dataset.from_tensor_slices(list(fullPthIds))
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  filenamepatterns_ds = fileIds_ds.map(lambda x: fileId2FnPattern(x,pathDir,channels), num_parallel_calls=AUTOTUNE)
  ########################################################################
  filenames = []
  labels = []
  scalars = []
  missing_file_idx = []
  missing_file_regexp = []
  keeped = np.ones(len(dfTimeseries), dtype=bool)
  # dfTimeseries = dfTimeseries.set_index('id')
  dfTimeseries = dfTimeseries.reset_index()  
  for idx,pattern in enumerate(filenamepatterns_ds):
    # image path retrieval
    try:
      #print(channels, timesteps)
      # print(pattern.numpy())
      files = [sorted(glob(pattern.numpy()[chanIdx]))[0] \
                for chanIdx in range(len(channels))]
      files = [f.decode("utf-8")  for f in files]
      if len(files)==0:
        raise(f'Missing File {pattern.numpy()}')
      filenames.append(files)

      label = dfTimeseries.loc[idx,[f'label_{offLabel}'  for offLabel in ts_off_label_hours]].values
      labels.append(label)
      if ts_off_scalar_hours is not None:
        scalar = dfTimeseries.loc[idx,[f'scalar_{offScalar}'  for offScalar in ts_off_scalar_hours]].values
        scalars.append(scalar)
      # if stricly_pos_raw_labels:
      #   if label < 0:
      #     label = 10*tf.keras.backend.epsilon()
      
    except Exception as e:
      #print(e)
      keeped[idx] = False
      missing_file_idx.append(idx)
      missing_file_regexp.append(pattern.numpy()[0])
  labels = np.array(labels)
  labels = labels.astype('float32')
  if ts_off_scalar_hours is not None:
    scalars = np.array(scalars)
    scalars = scalars.astype('float32')
  if strictly_pos_label:
    labels[labels<=0] = 0e-15
  
  print('------------------------------')
  print('labels.shape', labels.shape)
  print('------------------------------')
  
  # weighting -- NOT USED FOR FICAT
  # if weightByClass:
  actualWeights = {}
  if weightByClass:
    if len(labels.shape) < 2:
      labelWeightingCol = np.copy(labels)
    else:
      if weightOffLabIdx is None:
        labelWeightingCol = np.copy(labels[:,0])
      else:
        labelWeightingCol = np.copy(labels[:,weightOffLabIdx])
    print('labelCol', labelCol)
    print('classTresholds', classTresholds)
    weights = np.ones(len(labelWeightingCol))
    classes = list(classTresholds.keys())
    for cls in classes:
      print('CLASS', cls)
      clsIdxs = (labelWeightingCol>=classTresholds[cls][0]) & (labelWeightingCol<classTresholds[cls][1])
      print('len(labels)', len(labelWeightingCol))
      print('len(labels[clsIdxs])', len(labelWeightingCol[clsIdxs]))
      actualWeights[cls] = len(labelWeightingCol[clsIdxs]) / len(labelWeightingCol)
      print('actualWeights[cls', actualWeights[cls])
      weights[clsIdxs] = classWeights[cls] / actualWeights[cls]
    weights_ds = tf.data.Dataset.from_tensor_slices(weights)
    weights_ds = weights_ds.map(lambda x: tf.cast(x, dtype='float32'))
    
  # encoding
  if labelEncoder is not None:
    labels = np.fromiter((labelEncoder(x) for x in labels), dtype = 'float32')# labels.map(lambda x: labelEncoder(x))
  if scalarEncoder is not None and ts_off_scalar_hours is not None:
    scalars = np.fromiter((scalarEncoder(x) for x in scalars), dtype = 'float32')
    
  # tensorflow ds 
  # print(labels.shape)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  if ts_off_scalar_hours is not None:
    scalars_ds = tf.data.Dataset.from_tensor_slices(scalars)
    scalars_ds = scalars_ds.map(lambda x: tf.cast(x, dtype='float32'))
    
  if regression:
    labels_ds = labels_ds.map(lambda x: tf.cast(x, dtype='float32'))
  else:
    # TODO : make mutlilabel generic
    labels_ds = labels_ds.map(lambda x: tf.cast(x, tf.uint8))
    labels_ds = labels_ds.map(lambda x: tf.one_hot(x,num_classes))
    
  #######################################################################
  filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
  
  im = np.array(Image.open(filenames[0][0]))
  isGray = True if len(im.shape)==2 else False
  
  images_ds = filenames_ds.map(lambda x: parse_image(x,pathDir,gray2RGB, isGray, sepPosNeg), num_parallel_calls=AUTOTUNE) #.batch(batch_size)
  
  if shape3d:
    images_ds = images_ds.map(lambda x: tf.expand_dims(tf.transpose(x,[2,0,1]), axis=-1),num_parallel_calls=AUTOTUNE) # if no prior batching
  
  imHeight = im.shape[0]
  imWidth = im.shape[1]
  if img_size is not None:
    print('img_size', img_size)
    print('im.shape', im.shape)
    if crop_coord is not None:
      left, top, right, bottom = crop_coord
      imHeight = top - bottom
      imWidth = right - left
      images_ds = images_ds.map(lambda x: tf.image.crop_to_bounding_box(x,
                                                                        offset_height = top, 
                                                                        offset_width = left, 
                                                                        target_height = imHeight, 
                                                                        target_width = imWidth
                                                                        ) ,num_parallel_calls=AUTOTUNE)
    if imHeight != img_size[0] or imWidth != img_size[1]:
      if imHeight == imWidth and img_size[0] != img_size[1] and crop_coord is None:
        # default cropping for flare pole
        if img_size[1] > img_size[0]: #  width > height
          top = (img_size[1] + img_size[0]) / 2
          left = img_size[1]
        else:
          top = img_size[0]
          left = (img_size[1] + img_size[0]) / 2
        images_ds = images_ds.map(lambda x: tf.image.crop_to_bounding_box(x,
                                                                        offset_height = top, 
                                                                        offset_width = left, 
                                                                        target_height =  img_size[0], 
                                                                        target_width = img_size[1]
                                                                        ) ,num_parallel_calls=AUTOTUNE)
      images_ds = images_ds.map(lambda x: tf.image.resize(x,
                                                          size = (img_size[0], img_size[1]),
                                                          method=tf.image.ResizeMethod.BICUBIC, #BILINEAR,
                                                          preserve_aspect_ratio=True
                                                          ) ,num_parallel_calls=AUTOTUNE)
    #images_ds = images_ds.map(lambda x: tf.expand_dims(tf.transpose(x,[0,3,1,2]), axis=-1),num_parallel_calls=AUTOTUNE)
  if ts_off_scalar_hours is not None:
    if weightByClass:
      ds = tf.data.Dataset.zip((scalars_ds, images_ds, labels_ds, weights_ds))
      def structure_ds(a,b,c,d):
        return {
            'scalars': a,
            'image': b, 
        }, c, d
      ds = ds.map(structure_ds)
    else:
      ds = tf.data.Dataset.zip((scalars_ds, images_ds, labels_ds))
      def structure_ds(a,b,c):
        return {
            'scalars': a,
            'image': b, 
        }, c
      ds = ds.map(structure_ds)
  else:
    if weightByClass:
      ds = tf.data.Dataset.zip((images_ds, labels_ds, weights_ds))
    else:
      ds = tf.data.Dataset.zip((images_ds, labels_ds))

  ds = configure_for_performance(ds, batch_size, shuffle_buffer_size, shuffle, cache, prefetch, None, cachePath)
  dfTimeseries_updated = dfTimeseries[keeped].copy()
  return ds, missing_file_idx, missing_file_regexp, dfTimeseries_updated

def buildDS_persistant_MTS(
                dfTimeseries,
                samples,
                shiftSamplesByLabelOff = False,
                ts_off_label_hours   = [24], # offset from sample date
                ts_off_history_hours = [0], # offset from sample date
                labelEncoder      = None,
                encoderIsTf       = True,
                labelCol          = 'mpf',
                prefetch          = True,
                cache             = True,
                shuffle           = True,
                uncachedShuffBuff = 1000,
                regression        = False,
                num_classes       = None, 
                batch_size = 32,
                weightByClass = False,
                classTresholds = {'quiet': (0,1e-7), 'B':(1e-7,1e-6), 'C':(1e-6,1e-5), 'M':(1e-5,1e-4), 'X': (1e-4,np.inf)},
                classWeights = {'quiet': 0.2, 'B':0.2, 'C':0.2, 'M':0.2, 'X': 0.2},
                img_size = None, # for compatibility only
                weightOffLabIdx = None,
                dates2exclude = None,
                **kwargs
                ):
  if labelEncoder is not None and encoderIsTf:
    temp = labelEncoder
    labelEncoder = lambda x: temp(x).numpy()
  dfTimeseries = dfTimeseries.copy()
  samples = samples.copy()
  if type(ts_off_label_hours) not in [np.ndarray,list]:
    ts_off_label_hours = [ts_off_label_hours]
  if type(ts_off_history_hours) not in [np.ndarray,list]:
    ts_off_history_hours = [ts_off_history_hours]
    
  # offseting
  for offLabel in ts_off_label_hours:
    dfTimeseries[f'label_{offLabel}'] = dfTimeseries[labelCol].rolling(
            window = f'{offLabel}H',
            closed = 'right', # min_periods = int(scalar_lag)
            ).apply(lambda x: x[-1]).shift(freq = f'-{offLabel}H')#[:-int(offLabel/2)] 
    numna = dfTimeseries[f'label_{offLabel}'].isna().sum()
    print(f'WARNING : {numna} NaN (droped) for label at ts {offLabel}')  
  for offhistory in ts_off_history_hours:
    history_lag = -offhistory
    dfTimeseries[f'history_{offhistory}'] = dfTimeseries[labelCol].rolling(
        window = f'{history_lag}H',
        closed = 'both', # min_periods = int(history_lag)
        ).apply(lambda x: x[0])#[24*history_lag:] 
    numna = dfTimeseries[f'history_{offhistory}'].isna().sum()
    print(f'WARNING : {numna} NaN (droped) for input at ts {offhistory}')  
    
  dfTimeseries = dfTimeseries[int(np.max(np.abs(ts_off_history_hours))/2) : -int(np.max(ts_off_label_hours)/2)]  
  dfTimeseries.dropna(subset=[f'label_{offLabel}' for offLabel in ts_off_label_hours])
  dfTimeseries.dropna(subset=[f'history_{offhistory}' for offhistory in ts_off_history_hours])
      
    
  # offseting
  # dfTimeseries.index = dfTimeseries.index.shift(periods = -ts_off_label_hours, freq='H')
  # input_lag = - ts_off_history_hours
  # dfTimeseries['history'] = dfTimeseries[labelCol].rolling(window = f'{input_lag}H',
  #                                                  closed = 'right', # min_periods = int(input_lag)
  #                                                  ).apply(
  #                                                     lambda x: x[0]) # we remove first month in case of incomplete windows
  # dfTimeseries = dfTimeseries[int(ts_off_label_hours/2):-int(ts_off_label_hours/2)]
  # numna = dfTimeseries['history'].isna().sum()
  # print(f'WARNING : {numna} NaN (droped)')
  # dfTimeseries = dfTimeseries.dropna()

  # filtering on sample dates
  if samples is not None:
    if shiftSamplesByLabelOff: 
      # because the balance of the sample was made on the actual window values, not their foreccast-labels
      samples.index = samples.index + pd.DateOffset(hours= -ts_off_label_hours[0])
    dfTimeseries = dfTimeseries[dfTimeseries.index.isin(samples.index)]
    
  if dates2exclude is not None:
    dfTimeseries = dfTimeseries[~dfTimeseries.index.isin(dates2exclude)]

  # weighting
  if weightByClass:
    if weightOffLabIdx is None:
      labelWeightingCol = dfTimeseries[f'label_{ts_off_label_hours[0]}']
    else:
      labelWeightingCol = dfTimeseries[f'label_{ts_off_label_hours[weightOffLabIdx]}']
    classes = list(classTresholds.keys())
    actualWeights = {}
    weights = np.ones(len(labelWeightingCol))
    print('labelCol',labelCol)
    print('classTresholds',classTresholds)
    for cls in classes:
      print('CLASS', cls)
      clsIdxs = (labelWeightingCol>=classTresholds[cls][0]) & (labelWeightingCol<classTresholds[cls][1])
      print('len(labels)', len(labelWeightingCol))
      print('len(labels[clsIdxs])', len(labelWeightingCol[clsIdxs]))
      actualWeights[cls] = len(labelWeightingCol[clsIdxs]) / len(labelWeightingCol)
      print('actualWeights[cls', actualWeights[cls])
      weights[clsIdxs] = classWeights[cls] / actualWeights[cls]
    weights_ds = tf.data.Dataset.from_tensor_slices(weights)
    weights_ds = weights_ds.map(lambda x: tf.cast(x, dtype='float32'))
    
  # # encoding
  # if labelEncoder is not None:
  #   for offhistory in ts_off_history_hours:
  #     dfTimeseries[f'history_{offhistory}'] = dfTimeseries[f'history_{offhistory}'].apply(lambda x: labelEncoder(x))
  #   for offLabel in ts_off_label_hours:
  #     dfTimeseries[f'label_{offLabel}'] = dfTimeseries[f'label_{offLabel}'].apply(lambda x: labelEncoder(x))
  # # tensorflow ds
  # labels = dfTimeseries[[f'label_{offLabel}'  for offLabel in ts_off_label_hours]].values
  # inputs = dfTimeseries[[f'history_{offhistory}'  for offhistory in ts_off_history_hours]].values
  # labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  # inputs_ds = tf.data.Dataset.from_tensor_slices(inputs)
  
  # encoding
  labels = dfTimeseries[[f'label_{offLabel}'  for offLabel in ts_off_label_hours]].values
  inputs = dfTimeseries[[f'history_{offhistory}'  for offhistory in ts_off_history_hours]].values
  
  # if labelEncoder is not None:
  #   labels = labelEncoder(labels)
  #   inputs = labelEncoder(inputs)
  if labelEncoder is not None:
    labels = np.fromiter((labelEncoder(x) for x in labels), dtype = 'float32')# labels.map(lambda x: labelEncoder(x))
    inputs = np.fromiter((labelEncoder(x) for x in inputs), dtype = 'float32')

  # tensorflow ds
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  inputs_ds = tf.data.Dataset.from_tensor_slices(inputs)
  
  if not regression:
    labels_ds = labels_ds.map(lambda x: tf.cast(x, tf.uint8))
    labels_ds = labels_ds.map(lambda x: tf.one_hot(x,num_classes))
    inputs_ds = inputs_ds.map(lambda x: tf.cast(x, tf.uint8))
    inputs_ds = inputs_ds.map(lambda x: tf.one_hot(x,num_classes))
  else:
    labels_ds = labels_ds.map(lambda x: tf.cast(x, dtype='float32'))
    inputs_ds = inputs_ds.map(lambda x: tf.cast(x, dtype='float32'))

  if weightByClass:
    ds = tf.data.Dataset.zip((inputs_ds, labels_ds, weights_ds))
  else:
    ds = tf.data.Dataset.zip((inputs_ds, labels_ds))
  shuffle_buffer_size = uncachedShuffBuff
  ds = configure_for_performance(ds, batch_size, shuffle_buffer_size, shuffle, cache, prefetch)
  # dfTimeseries_updated = dfTimeseries[keeped]
  dfTimeseries_updated = dfTimeseries.copy()
  return ds, [], [], dfTimeseries_updated

def get_anomalies_dates(anomaliesGradeFolder, wavelengths, qualityTreshold = 1):
  
  anomaliesDates = {}
  for idx, w in enumerate(wavelengths):
    anommaliesPath = anomaliesGradeFolder/f'{w}_anomalies_notes.csv'
    anomaliesDates[w] = read_Dataframe_With_Dates(anommaliesPath)
    anomaliesDates[w] = anomaliesDates[w][anomaliesDates[w]['grade'] < qualityTreshold]
    if idx==0:
      dates2exclude = anomaliesDates[w].index.values
    else:
      dates2exclude = np.concatenate([dates2exclude, anomaliesDates[w].index.values])
  dates2exclude.shape, np.unique(dates2exclude).shape

  dates2exclude = np.sort(np.unique(dates2exclude))
  
  return dates2exclude

def builDS_ts_feature(
  pathDir,
  channels,
  batch_size,
  samples, # sample dates
  dfTimeseries,
  dfTimeseriesLabel = None, 
  shiftSamplesByLabelOff = False,
  ts_off_label_hours = 24*np.arange(0.5,6.5,0.5),
  ts_off_scalar_hours = None, 
  labelCol          = ['sw_v'],
  scalarCol         = None,
  prefetch          = True,
  cache             = True,
  cachePath         = '', # keep string empty for ram caching
  shuffle           = True,
  uncachedShuffBuff = 1000,
  img_size          = None,
  crop_coord        = None,
  num_classes       = 2,
  gray2RGB          = True,
  sepPosNeg         = False,
  shape3d           = False,
  regression        = False,
  labelEncoder      = None,
  encoderIsTf       = True,
  scalarEncoder     = None,
  weightByClass     = False,
  weightOffLabIdx   = 0,
  # needed only if weightByClass is True, class are defined relatively to labelCol
  classTresholds    = {'quiet': (0,1e-7), 
                       'B':(1e-7,1e-6), 
                       'C':(1e-6,1e-5), 
                       'M':(1e-5,1e-4), 
                       'X':(1e-4,np.inf)},
  classWeights      = {'quiet': 0.20, 
                       'B':0.20, 
                       'C':0.20, 
                       'M':0.20, 
                       'X':0.20},
  strictly_pos_label = True,
  dates2exclude = None,
  **kwargs # bacckward compt
):
  if scalarCol is None:
    scalarCol = labelCol
  if type(scalarCol) ==  str:
    scalarCol = [scalarCol]
  if type(labelCol) ==  str:
    labelCol = [labelCol]
  if ts_off_scalar_hours is None:
    ts_off_scalar_hours =  ts_off_label_hours
  if dfTimeseriesLabel is None:
    dfTimeseriesLabel = dfTimeseries.copy()
  if labelEncoder is not None and encoderIsTf:
    temp = labelEncoder
    labelEncoder = lambda x: temp(x).numpy()
  dfTimeseries = dfTimeseries.copy()
  if samples is not None:
    samples = samples.copy()
  if scalarCol is None:
    scalarCol = labelCol
  if type(ts_off_label_hours) not in [np.ndarray,list]:
    ts_off_label_hours = [ts_off_label_hours]
  if ts_off_scalar_hours is not None:
    if type(ts_off_scalar_hours) not in [np.ndarray,list]:
      ts_off_scalar_hours = [ts_off_scalar_hours]
  '''
  '''
  # 
  if len(channels)>2: 
    gray2RGB=False
  if isinstance(pathDir,pathlib.PosixPath): 
    pathDir = pathDir.as_posix()
    
  # offseting
  # if shiftTsByLabOff:
  for offLabel in ts_off_label_hours:
    for labCol in labelCol:
      dfTimeseriesLabel[f'label_{labCol}_{offLabel}'] = dfTimeseriesLabel[labCol].rolling(
              window = f'{offLabel}H',
              closed = 'right', # min_periods = int(scalar_lag)
              ).apply(lambda x: x[-1]).shift(freq = f'-{offLabel}H')#[:-int(offLabel/2)] 
      numna = dfTimeseriesLabel[f'label_{labCol}_{offLabel}'].isna().sum()
      print(f'WARNING : {numna} NaN (droped) for label {labCol} at ts {offLabel}')
    
  for offScalar in ts_off_scalar_hours:
    # WARNING : offScalar is expeccted negative or null
    scalar_lag = -offScalar #- int(offScalar//24)
    for scCol in scalarCol:
      dfTimeseries[f'scalar_{scCol}_{offScalar}'] = dfTimeseries[scCol].rolling(
          window = f'{scalar_lag}H',
          closed = 'both', # min_periods = int(scalar_lag)
          ).apply(lambda x: x[0])#[24*scalar_lag:] 
    
  # feature - label merging
  dfTimeseries = pd.concat([dfTimeseries[[col for col in dfTimeseries.columns if 'scalar' in col]], 
                            dfTimeseriesLabel[[col for col in dfTimeseriesLabel.columns if 'label' in col]]
                            ], 
                           axis = 1, join='inner')
  
  if ts_off_scalar_hours is not None:
    startIdx = int(np.max(np.abs(ts_off_scalar_hours))/2)
  else:
    startIdx = 0
  dfTimeseries = dfTimeseries[startIdx : -int(np.max(ts_off_label_hours)/2)]  
  for labCol in labelCol:
    dfTimeseries = dfTimeseries.dropna(subset=[f'label_{labCol}_{offLabel}' for offLabel in ts_off_label_hours])
      
  # fiiltering on sample dates
  if samples is not None:
    if shiftSamplesByLabelOff:
      print('Samples shiiftng done')
      # use when the balance of the sample is made on the actual window values, not their foreccast-labels
      samples.index = samples.index + pd.DateOffset(hours= -ts_off_label_hours[0])
    dfTimeseries = dfTimeseries[dfTimeseries.index.isin(samples.index)]
    
  if dates2exclude is not None:
    dfTimeseries = dfTimeseries[~dfTimeseries.index.isin(dates2exclude)]

  if not cache and shuffle:
    # shuffle sumplement for small buffer
    dfTimeseries = dfTimeseries.sample(frac=1)
    shuffle_buffer_size = len(dfTimeseries)
  else:
    shuffle_buffer_size = uncachedShuffBuff

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  ########################################################################
  labels = []
  scalars = []
  
  labels  = dfTimeseries[[col for col in dfTimeseries.columns if 'label' in col]].values
  
  # scalars = dfTimeseries[[col for col in dfTimeseries.columns if 'scalar' in col]].values
  scalars = np.stack([dfTimeseries[[col for col in dfTimeseries.columns if f'scalar_{scCol}' in col]].values for scCol in scalarCol], axis = -1)

  labels = np.array(labels)
  labels = labels.astype('float32')
  if ts_off_scalar_hours is not None:
    scalars = np.array(scalars)
    scalars = scalars.astype('float32')
  if strictly_pos_label:
    labels[labels<=0] = 0e-15
  
  print('------------------------------')
  print('labels.shape', labels.shape)
  print('------------------------------')
  
  # weighting -- NOT USED FOR FICAT
  # if weightByClass:
  actualWeights = {}
  if weightByClass:
    if len(labels.shape) < 2:
      labelWeightingCol = np.copy(labels)
    else:
      if weightOffLabIdx is None:
        labelWeightingCol = np.copy(labels[:,0])
      else:
        labelWeightingCol = np.copy(labels[:,weightOffLabIdx])
    print('labelCol', labelCol)
    print('classTresholds', classTresholds)
    weights = np.ones(len(labelWeightingCol))
    classes = list(classTresholds.keys())
    for cls in classes:
      print('CLASS', cls)
      clsIdxs = (labelWeightingCol>=classTresholds[cls][0]) & (labelWeightingCol<classTresholds[cls][1])
      print('len(labels)', len(labelWeightingCol))
      print('len(labels[clsIdxs])', len(labelWeightingCol[clsIdxs]))
      actualWeights[cls] = len(labelWeightingCol[clsIdxs]) / len(labelWeightingCol)
      print('actualWeights[cls', actualWeights[cls])
      weights[clsIdxs] = classWeights[cls] / actualWeights[cls]
    weights_ds = tf.data.Dataset.from_tensor_slices(weights)
    weights_ds = weights_ds.map(lambda x: tf.cast(x, dtype='float32'))
    
  # encoding
  if labelEncoder is not None:
    labels = np.fromiter((labelEncoder(x) for x in labels), dtype = 'float32')# labels.map(lambda x: labelEncoder(x))
  if scalarEncoder is not None and ts_off_scalar_hours is not None:
    scalars = np.fromiter((scalarEncoder(x) for x in scalars), dtype = 'float32')
    
  # tensorflow ds 
  # print(labels.shape)
  labels_ds = tf.data.Dataset.from_tensor_slices(labels)
  if ts_off_scalar_hours is not None:
    scalars_ds = tf.data.Dataset.from_tensor_slices(scalars)
    scalars_ds = scalars_ds.map(lambda x: tf.cast(x, dtype='float32'))
    
  if regression:
    labels_ds = labels_ds.map(lambda x: tf.cast(x, dtype='float32'))
  else:
    # TODO : make mutlilabel generic
    labels_ds = labels_ds.map(lambda x: tf.cast(x, tf.uint8))
    labels_ds = labels_ds.map(lambda x: tf.one_hot(x,num_classes))
    
  #######################################################################

    #images_ds = images_ds.map(lambda x: tf.expand_dims(tf.transpose(x,[0,3,1,2]), axis=-1),num_parallel_calls=AUTOTUNE)
  if weightByClass:
    ds = tf.data.Dataset.zip((scalars_ds, labels_ds, weights_ds))
    def structure_ds(a,c,d):
      return {
          'scalars': a,
      }, c, d
    ds = ds.map(structure_ds)
  else:
    ds = tf.data.Dataset.zip((scalars_ds, labels_ds))
    def structure_ds(a,c):
      return {
          'scalars': a,
      }, c
    ds = ds.map(structure_ds)


  ds = configure_for_performance(ds, batch_size, shuffle_buffer_size, shuffle, cache, prefetch, None, cachePath)
  dfTimeseries_updated = dfTimeseries.copy()
  return ds, [], [], dfTimeseries_updated

# def builDS_image(pathDir,
#   channels,
#   batch_size,
#   dfTimeseries,
#   samples, # sample dates
#   epochs,
#   shiftSamplesByLabelOff = True,
#   shiftTsByLabOff   = True,
#   ts_off_label_hours= 24,
#   labelCol          = 'label',
#   prefetch          = True,
#   cache             = True,
#   shuffle           = True,
#   uncachedShuffBuff = 1000,
#   img_size          = None,
#   num_classes       = 2,
#   gray2RGB          = True,
#   chnWiseNormalize  = False,
#   pathNormFile      = '',
#   shape3d           = False,
#   regression        = False,
#   labelEncoder      = None,
#   weightByClass     = False,
#   classTresholds    = {'quiet': (0,1e-7), 
#                        'B':(1e-7,1e-6), 
#                        'C':(1e-6,1e-5), 
#                        'M':(1e-5,1e-4), 
#                        'X':(1e-4,np.inf)},
#   classWeights      = {'quiet': 0.20, 
#                        'B':0.20, 
#                        'C':0.20, 
#                        'M':0.20, 
#                        'X':0.20},
# **kwargs # bacckward compt
# ):
#   '''
#   Generate a tensorflow dataset of images and features
#   Parameters
#   ----------
#   channels : `list` of `string` or `int`
#     images channel ID to read and assemble from directory 'pathDir'
    
#   dfTimeseries : DataFrame
#     continuous timeseries containing labels and eventual features,
#     features can be added in this dataset only from dfTimeseries
    
#   samples : DataFrame
#     DataFrame where the index contain the date of the samples to be used in the dataset
    
#   shiftTsByLabOff : bool, optional
#     shift the 'dfTimeseries' dates of 'ts_off_label_hours' hours, 
#     needed if dfTimeseries is given at feature levels, 
#     i.e. values at date D gives feature of timestep 0 for date D and not labels,
#     in the case of flare forecasting the features of timestep 0 
#     characterize the time-window [D-windowSize ; D[,
#     while the labels values refer to [D ; D + windowSize[,
#     default to true
    
#   shiftSamplesByLabelOff : bool, optional
#     shift the 'sample' dates of 'ts_off_label_hours' hours, 
#     needed if the sample dates where computed at feature level rather than featur,
#     default to true

#   gray2RGB : `bool`, optional
#     only for one channel images (i.e. len(channel)==len(timesteps)==1)
#     convert the resulting 1 channel image into a 3 channel one by repeating it
#     usefull to use premade NN on 1 channel images

#   Returns
#   -------
#   `tf.data.Dataset`
#     a tensorflow dataset where an image is of the shape : [img_size,img_size,len(channels)+len(timesteps)]
#     the channels or organized as follow :
#     channel_1
#     ...
#     channel_n 
#   '''
#   # 
#   if len(channels)>2: 
#     gray2RGB=False
#   if isinstance(pathDir,pathlib.PosixPath): 
#     pathDir = pathDir.as_posix()

#   # offseting
#   if shiftTsByLabOff:
#     dfTimeseries.index = dfTimeseries.index.shift(periods = -ts_off_label_hours, freq='H')
#     dfTimeseries = dfTimeseries[int(ts_off_label_hours/2):-int(ts_off_label_hours/2)]
#   numna = dfTimeseries[labelCol].isna().sum()
#   print(f'WARNING : {numna} NaN (droped)')
#   dfTimeseries = dfTimeseries.dropna()
#   # fiiltering on sample dates
#   if samples is not None:
#     if shiftSamplesByLabelOff:
#       print('Samples shiiftng done')
#       # because the balance of the sample was made on the actual window values, not their foreccast-labels
#       samples.index = samples.index + pd.DateOffset(hours= -ts_off_label_hours)
#     dfTimeseries = dfTimeseries[dfTimeseries.index.isin(samples.index)]

#   if not cache and shuffle:
#     # necessary shuffle sumplement with small buffer
#     dfTimeseries = dfTimeseries.sample(frac=1)
#     shuffle_buffer_size = len(dfTimeseries)
#   else:
#     shuffle_buffer_size = uncachedShuffBuff

#   dfTimeseries = dfTimeseries.reset_index()
#   dfTimeseries['id'] = dfTimeseries['timestamp'].apply(lambda x: x.strftime('%Y%m%d_%H%M'))
#   dfTimeseries['pth'] = dfTimeseries['timestamp'].apply(lambda x: x.strftime('%Y/%m/%d'))
#   dfTimeseries = dfTimeseries.set_index('timestamp')
#   fullPthIds = dfTimeseries[['pth','id']].apply(lambda x: x['pth']+'/'+x['id'],axis=1)
#   fileIds_ds = tf.data.Dataset.from_tensor_slices(list(fullPthIds))
#   AUTOTUNE = tf.data.experimental.AUTOTUNE
#   filenamepatterns_ds = fileIds_ds.map(lambda x: fileId2FnPattern(x,pathDir,channels), num_parallel_calls=AUTOTUNE)
#   ########################################################################
#   filenames = []
#   labels = []
#   missing_file_idx = []
#   missing_file_regexp = []
#   keeped = np.ones(len(dfTimeseries), dtype=bool)
#   # dfTimeseries = dfTimeseries.set_index('id')
#   dfTimeseries = dfTimeseries.reset_index()
#   for idx,pattern in enumerate(filenamepatterns_ds):
#     # image path retrieval
#     try:
#       #print(channels, timesteps)
#       files = [sorted(glob(pattern.numpy()[chanIdx]))[0] \
#                 for chanIdx in range(len(channels))]
#       files = [f.decode("utf-8")  for f in files]
#       filenames.append(files)
#       label = dfTimeseries.loc[idx,labelCol]
#       # if stricly_pos_raw_labels:
#       #   if label < 0:
#       #     label = 10*tf.keras.backend.epsilon()
#       labels.append(label)
#     except Exception as e:
#       #print(e)
#       keeped[idx] = False
#       missing_file_idx.append(idx)
#       missing_file_regexp.append(pattern.numpy()[0])
#   labels = np.array(labels)
#   labels[labels<=0] = 0
#   # weighting
#   # if weightByClass:
#   print('labelCol', labelCol)
#   print('classTresholds', classTresholds)
#   classes = list(classTresholds.keys())
#   actualWeights = {}
#   if weightByClass:
#     weights = np.ones(len(labels))
#     for cls in classes:
#       print('CLASS', cls)
#       clsIdxs = (labels>=classTresholds[cls][0]) & (labels<classTresholds[cls][1])
#       print('len(labels)', len(labels))
#       print('len(labels[clsIdxs])', len(labels[clsIdxs]))
#       actualWeights[cls] = len(labels[clsIdxs]) / len(labels)
#       print('actualWeights[cls', actualWeights[cls])
#       weights[clsIdxs] = classWeights[cls] / actualWeights[cls]
#     weights_ds = tf.data.Dataset.from_tensor_slices(weights)
#     weights_ds = weights_ds.map(lambda x: tf.cast(x, dtype='float32'))
#   # encoding
#   if labelEncoder is not None:
#     labels = np.fromiter((labelEncoder(x) for x in labels), dtype = 'float32')# labels.map(lambda x: labelEncoder(x))
#   # tensorflow ds
#   labels_ds = tf.data.Dataset.from_tensor_slices(labels)
#   if regression:
#     labels_ds = labels_ds.map(lambda x: tf.cast(x, dtype='float32'))
#   else:
#     labels_ds = labels_ds.map(lambda x: tf.cast(x, tf.uint8))
#     labels_ds = labels_ds.map(lambda x: tf.one_hot(x,num_classes))
#   #######################################################################
#   filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
#   im = np.array(Image.open(filenames[0][0]))
#   isGray = True if len(im.shape)==2 else False
  
#   if chnWiseNormalize:
#     duration = time.time()
#     filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
#     pixstat = pd.read_csv(pathNormFile).set_index('channel')
#     tfMeans = tf.constant([pixstat.loc[str(channel)]['mean_wg'] for channel in channels], dtype='float32')
#     tfStd = tf.constant([pixstat.loc[str(channel)]['std_wg'] for channel in channels], dtype='float32')
#     @tf.autograph.experimental.do_not_convert
#     def tf_normaliser(x):
#       return parse_image_normalize_tfMode(x,gray2RGB, tfMeans, tfStd)
#     images_ds = filenames_ds.map(tf_normaliser, num_parallel_calls=AUTOTUNE)
#     duration = time.time() - duration
#     print(f'Normalisation took {duration//60:0>2.0f}m{duration%60:0>2.0f}s during dataset instantiation')
#   else:
#     images_ds = filenames_ds.map(lambda x: parse_image(x,pathDir,gray2RGB, isGray), num_parallel_calls=AUTOTUNE) #.batch(batch_size)
  
#   if shape3d:
#     images_ds = images_ds.map(lambda x: tf.expand_dims(tf.transpose(x,[2,0,1]), axis=-1),num_parallel_calls=AUTOTUNE) # if no prior batching
  
#   if img_size is not None:
#     print('img_size', img_size)
#     print('im.shape', im.shape)
#     if im.shape[0] != img_size[0] or im.shape[1] != img_size[1]:
#       images_ds = images_ds.map(lambda x: tf.image.resize(x,
#                                                           size = (img_size[0], img_size[1]),
#                                                           method=tf.image.ResizeMethod.BICUBIC, #BILINEAR,
#                                                           preserve_aspect_ratio=True
#                                                           ) ,num_parallel_calls=AUTOTUNE)
#     #images_ds = images_ds.map(lambda x: tf.expand_dims(tf.transpose(x,[0,3,1,2]), axis=-1),num_parallel_calls=AUTOTUNE)

#   if weightByClass:
#     ds = tf.data.Dataset.zip((images_ds, labels_ds, weights_ds))
#   else:
#     ds = tf.data.Dataset.zip((images_ds, labels_ds))

#   ds = configure_for_performance(ds, batch_size, shuffle_buffer_size, shuffle, cache, prefetch, epochs)
#   dfTimeseries_updated = dfTimeseries[keeped]
#   return ds, missing_file_idx, missing_file_regexp, dfTimeseries_updated

# def buildDS_persistant(num_classes,
#                 dfTimeseries,
#                 samples,
#                 shiftSamplesByLabelOff = True,
#                 ts_off_label_hours   = 24, # offset from sample date
#                 ts_off_history_hours = -24, # offset from label date
#                 labelEncoder = None,
#                 historyEncoder = None,
#                 labelCol          = 'mpf',
#                 prefetch          = True,
#                 cache             = True,
#                 shuffle           = True,
#                 uncachedShuffBuff = 1000,
#                 regression        = False,
#                 epochs            = 1,
#                 batch_size = 32,
#                 weightByClass = False,
#                 classTresholds = {'quiet': (0,1e-7), 'B':(1e-7,1e-6), 'C':(1e-6,1e-5), 'M':(1e-5,1e-4), 'X': (1e-4,np.inf)},
#                 classWeights = {'quiet': 0.2, 'B':0.2, 'C':0.2, 'M':0.2, 'X': 0.2},
#                 img_size = None # for compatibility only
#                 ):
#   if historyEncoder is None: historyEncoder = labelEncoder

    
#   # offseting
#   dfTimeseries.index = dfTimeseries.index.shift(periods = -ts_off_label_hours, freq='H')
#   input_lag = - ts_off_history_hours
#   dfTimeseries['history'] = dfTimeseries[labelCol].rolling(window = f'{input_lag}H',
#                                                    closed = 'right', # min_periods = int(input_lag)
#                                                    ).apply(
#                                                       lambda x: x[0]) # we remove first month in case of incomplete windows
#   dfTimeseries = dfTimeseries[int(ts_off_label_hours/2):-int(ts_off_label_hours/2)]
#   numna = dfTimeseries['history'].isna().sum()
#   print(f'WARNING : {numna} NaN (droped)')
#   dfTimeseries = dfTimeseries.dropna()

#   # filtering on sample dates
#   if samples is not None:
#     if shiftSamplesByLabelOff:
#       # because the balance of the sample was made on the actual window values, not their foreccast-labels
#       samples.index = samples.index + pd.DateOffset(hours= -ts_off_label_hours)
#     dfTimeseries = dfTimeseries[dfTimeseries.index.isin(samples.index)]

#   # weighting
#   if weightByClass:
#     classes = list(classTresholds.keys())
#     actualWeights = {}
#     # labels = np.array(dfTimeseries[labelCol].tolist())
#     labels = dfTimeseries[labelCol].values
#     labels[labels<=0] = 0
#     weights = np.ones(len(labels))
#     print('labelCol',labelCol)
#     print('classTresholds',classTresholds)
#     for cls in classes:
#       print('CLASS', cls)
#       clsIdxs = (labels>=classTresholds[cls][0]) & (labels<classTresholds[cls][1])
#       print('len(labels)', len(labels))
#       print('len(labels[clsIdxs])', len(labels[clsIdxs]))
#       actualWeights[cls] = len(labels[clsIdxs]) / len(labels)
#       print('actualWeights[cls', actualWeights[cls])
#       weights[clsIdxs] = classWeights[cls] / actualWeights[cls]
#     weights_ds = tf.data.Dataset.from_tensor_slices(weights)
#     weights_ds = weights_ds.map(lambda x: tf.cast(x, dtype='float32'))

#   # encoding
#   if historyEncoder is not None:
#     dfTimeseries['history'] = dfTimeseries['history'].apply(lambda x: historyEncoder(x))
#   if labelEncoder is not None:
#     dfTimeseries[labelCol] = dfTimeseries[labelCol].apply(lambda x: labelEncoder(x))

#   # tensorflow ds
#   labels_ds = tf.data.Dataset.from_tensor_slices(dfTimeseries[labelCol].tolist())
#   flareHistoPreds_ds = tf.data.Dataset.from_tensor_slices(dfTimeseries['history'].tolist())
#   if not regression:
#     labels_ds = labels_ds.map(lambda x: tf.cast(x, tf.uint8))
#     labels_ds = labels_ds.map(lambda x: tf.one_hot(x,num_classes))
#   else:
#     labels_ds = labels_ds.map(lambda x: tf.cast(x, dtype='float32'))
#   flareHistoPreds_ds = flareHistoPreds_ds.map(lambda x: tf.cast(x, dtype='float32'))

#   if weightByClass:
#     ds = tf.data.Dataset.zip((flareHistoPreds_ds, labels_ds, weights_ds))
#   else:
#     ds = tf.data.Dataset.zip((flareHistoPreds_ds, labels_ds))
#   shuffle_buffer_size = uncachedShuffBuff
#   ds = configure_for_performance(ds, batch_size, shuffle_buffer_size, shuffle, cache, prefetch, epochs)
#   # dfTimeseries_updated = dfTimeseries[keeped]
#   return ds, [], [], []
