"""
Functions utilities  training/cv loop
"""

import os
from glob import glob
from pathlib import Path
from functools import reduce
import dill as pickle
import datetime
import numpy as np
import pandas as pd

import tensorflow as tf

from config import * # path to sundl
from sundl.models import ModelInstantier
from sundl.utils.flare.thresholds import mpfTresh, totehTresh
from sundl.constants import MIN_METRICS

class ModelInstantier2(ModelInstantier):
  def __init__(self,
    buildModelFunction,
    buildModelParams,
    buildDsFunction,
    buildDsParams,
    name,
    # fullNameFunc=None,
    savedPredictionModel = False,
    cls = None,
    weightStrategy = None,
    foldTag = '',
    extraNameTag = None
    ):
    super().__init__(buildModelFunction = buildModelFunction,
                     buildModelParams = buildModelParams,
                     buildDsFunction = buildDsFunction,
                     buildDsParams = buildDsParams,
                     name = name,
                    #  fullNameFunc = None,
                     savedPredictionModel = savedPredictionModel,
                     cls = cls
                     )
    self.weightStrategy = weightStrategy
    self.foldTag = foldTag
    self.extraNameTag = extraNameTag
    
  def _optimTag(self):
    optConf = self.buildModelParams['optimizer']
    name = optConf['name'][0] + optConf['name'][-1]
    lr = optConf['learning_rate']
    lr = f'{lr:.0e}'
    lr = f'{lr[0]}e{lr[-1]}'
    if 'weight_decay' in optConf.keys():
      wd = optConf['weight_decay']
      wd = f'{wd:.0e}'
      wd = f'D{wd[0]}e{wd[-1]}'
    else:
      wd = ''
    optimTag = name + lr + wd
    optimTag
    return optimTag
    
  def _parammetersTag(self):
    tag = ''
    if 'patches_size' in self.buildModelParams.keys():
      tag += '_PT'
      imSize = self.buildModelParams['img_size']
      ptSize = self.buildModelParams['patches_size']
      num_patches = int(imSize[0]*imSize[1] / (ptSize[0]*ptSize[1]))
      tag += f'x{num_patches}'
      if 'includeInterPatches' in self.buildModelParams.keys():
        if self.buildModelParams['includeInterPatches']:
          tag += f'xIntrPtc'
    else:
      tag += '_FD'
    if 'pretainedWeight' in self.buildModelParams.keys():
      if  self.buildModelParams['pretainedWeight'] is False:
        tag += '_FlTr'
      else:
        if 'unfreeze_top_N' in self.buildModelParams.keys():
          if  self.buildModelParams['unfreeze_top_N'] is not None:
            tag += '_RtdX'
            tag += str(self.buildModelParams['unfreeze_top_N'])
          else:
            tag += '_Ptrd'
    else:
      tag += '_O'
    if self.weightStrategy is not None:
      tag += f'_{self.weightStrategy}'
    else:
      tag += '_NoWght'
    if self.foldTag != '':
      tag += f'x{self.foldTag}'
    if 'optimizers' in self.buildModelParams.keys():
      tag += f'_{self._optimTag()}'
    if self.extraNameTag is not None:
      tag += f'_{self.extraNameTag}'
    return  tag

  def fullNameFunc(self, channels, h): 
    fullname = f'{self.name}_'
    fullname += reduce(lambda x,y: f'{x}x{y}',
                       [f'{channel:0>4}' for channel in channels]) 
    fullname += f'_{h}'
    fullname += '_' + self._parammetersTag()
    return fullname

def setUpResultFolder(models, 
                      pathRes,
                      metrics,
                      continuingFolder = None, 
                      newFolder = None,
                      imgSize = None,
                      cv_K = None,
                      epochs = None,
                      batchSize = None,
                      saveModel = False
                      ):
  if pathRes != str:
    pathRes = pathRes.as_posix()
  if continuingFolder is None:
    #creating a new result folder
    
    if newFolder is not None:
      folderTag = newFolder
    else:
      folderTag = reduce(lambda x,y:x+'x'+y, sorted(set([el[0].name for el in models])))
    
    cvTag = 'xCV'+f'{cv_K:0>2}' if cv_K is not None else ''
    imgSize_tag = reduce(lambda x,y : str(x)+'x'+str(y),imgSize) if imgSize is not None else ''
    epochsTag = f'x{epochs:0>3}' if epochs is not None else ''
    batchTag = f'x{batchSize:0>3}' if batchSize is not None else ''
   
    basedir = pathRes + f'/{folderTag}_{imgSize_tag}{epochsTag}{batchTag}{cvTag}_' + datetime.datetime.now().strftime('%Y_%m_%d__')
    listTrainDir = glob(basedir+'*')
    resDir = basedir + str(len(listTrainDir))
    os.mkdir(resDir)
    
    if cv_K is not None:
      os.mkdir(resDir+f'/training_folds')

    if saveModel:
      modelDir = resDir + '/models'
      os.mkdir(modelDir)
      mtcDict ={}
      for m in metrics:
        mtcDict[m.name] = m
      with open(resDir + '/models/metrics.pkl', 'wb') as f1:
        pickle.dump(mtcDict, f1)
    testPredDir =  F_PATH_PREDS(Path(resDir))
    os.mkdir(testPredDir)
      
    # Creating log file
    full_name_combs  = [model.fullNameFunc(channels,h) for model, channels, h in models]
    log = pd.DataFrame({'model':full_name_combs, 'status': np.zeros(len(full_name_combs),dtype=int), 'duration':  np.zeros(len(full_name_combs),dtype=str)})
    log = log.set_index('model')
    log.to_csv(resDir + '/log.csv')

  else:
    # Continuig an existing folder
    resDir = pathRes + f'/{continuingFolder}'
    modelDir = resDir + '/models'
    if saveModel:
      # with open(resDir + '/models/metrics.pkl', 'rb') as f1:
      #     mtcDict = pickle.load(f1)
      mtcDict = {}
      for m in metrics:
        mtcDict[m.name] = m
    log = pd.read_csv(resDir + '/log.csv')
    #full_name_combs  = [model.name + '_' + str(len(timesteps))+'ts_'+reduce(lambda x,y:x+'x'+y,[f'{channel:0>4}'[0:4] for channel in channels]) for model, channels, timesteps in models]
    full_name_combs  = [model.fullNameFunc(channels,h) for model, channels, h in models]
    new_combs = [comb for comb in full_name_combs if comb not in log['model'].values]
    log_new = pd.DataFrame({'model':new_combs, 'status': np.zeros(len(new_combs),dtype=int), 'duration':  np.zeros(len(new_combs),dtype=str)})
    log = log.append(log_new).set_index('model')
    log.to_csv(resDir + '/log.csv')
  return log, resDir, modelDir, mtcDict
    

def conditionalHyperParameters(modelInstantiater, h, save_thdS, weightCollection):
  binCls = modelInstantiater.cls
  save_thd = save_thdS[binCls]
  labelCol = modelInstantiater.buildDsParams['labelCol']
  print('\nlabelCol : ', labelCol)
  print('window_h : ', h)
  if modelInstantiater.weightStrategy is not None:
    classWeights = weightCollection[modelInstantiater.weightStrategy]
  else:
    classWeights = None
  if labelCol == 'mpf':
    classTresholds = mpfTresh
  elif labelCol == 'toteh':
    classTresholds = totehTresh[h]
  encoder = lambda flux : float(flux >= classTresholds[binCls][0])
  return save_thd, labelCol, binCls, classWeights, classTresholds, encoder

def cvInit(full_name_comb, df_train, df_val, save_model, modelDir, frac_sample, cv_K, kf, sampVal = False):
  dfSamples_train = df_train.copy()
  dfSamples_val = df_val.copy()
  if frac_sample is not None:
    dfSamples_train = dfSamples_train.sample(frac = frac_sample, random_state=49)
    if sampVal:
      dfSamples_val = dfSamples_val.sample(frac = frac_sample, random_state=49)
  print(f'#TRAIN : {len(dfSamples_train)}')
  print(f'#VAL   : {len(dfSamples_val)}')
  if save_model:
    modelDirSub = modelDir + f'/{full_name_comb}'
    if cv_K is not None:
      modelDirSub = modelDirSub + f'_fd{kf:0>3d}'
  
  
def trainConstantModel(dsTrain, dsVal, model, modelInstantiater, epochs, weightByClass, save_model, modelDirSub):
  """
  By definition constant model are not trainable
  Their training consist of simple evaluation on train and val
  for the sake of comparison with non constant models
  """
  print('Histo model on train : ')
  if weightByClass:
    x, y, w = None, None, None
    # evaluate doesn't work normally on ds with weghts
    # the weight argument need to be extended by one dimension
    for a, b, c in dsTrain.take(len(dsTrain)):
      if x is None:
        x, y, w = a, b, c
      else:
        x = tf.keras.layers.Concatenate(axis=0)([x, a])
        y = tf.keras.layers.Concatenate(axis=0)([y, b])
        w = tf.keras.layers.Concatenate(axis=0)([w, c])
    w = tf.expand_dims(w, axis=-1)
    train = model.evaluate(x=x,y=y,sample_weight=w)
  else:
    train = model.evaluate(dsTrain)
  print('Histo model on val : ')
  val = model.evaluate(dsVal)
  tfMetrics = model.metrics
  for i in range(len(tfMetrics)):
    if tfMetrics[i].name in ['f1','F1']:
      train[i] = train[i][1]
      val[i] = val[i][1]
  historyData = {tfMetrics[i].name : train[i]*np.ones(epochs) for i in range(len(tfMetrics))}
  historyData.update({'val_'+tfMetrics[i].name : val[i]*np.ones(epochs)  for i in range(len(tfMetrics))})
  if save_model:
    pathConfigModel = modelDirSub + f'_config.pkl'
    modelInstantiater.saveConfig(pathConfigModel)
    model.save(modelDirSub)
  return historyData 

def printTrainingResults(historyData, cat = ['']):
  for cat in ['']:
    print('')
    for m in historyData.keys():
      if m[:2] != 'va':
        print(f'Train {m}{cat} : ', historyData[f'{m}{cat}'])
    print('')
    for m in historyData.keys():
      if m[:2] == 'va':
        print(f'Val   {m}{cat} : ', historyData[f'{m}{cat}'])
    
def saveTrainingResults(resDir, res, best, bestCVCrossEpoch, full_name_comb, cv_K):
  metrics = res[full_name_comb][0].columns
  #===================================================
  # CV SPECIFIC
  if cv_K is not None:
    dfRes = res[full_name_comb][0]
    tmp = {'model': full_name_comb}
    for col in [col for col in metrics]:
      # feeling nan arrays with 0 --> find better? (not relevant for all metrics)
      mtc = np.array([res[full_name_comb][kFold][col].fillna(0).values for kFold in range(cv_K)])
      dfRes[col] = np.mean(mtc,axis=0)
      dfRes[col+'_std'] = np.std(mtc,axis=0)
      dfRes[col+'_min'] = np.min(mtc,axis=0)
      dfRes[col+'_max'] = np.max(mtc,axis=0)
      if col in MIN_METRICS:
        mtcCVCE = np.min(mtc,axis=1)
        mtcCVCE_epcs = reduce(lambda x,y: str(x)+'x'+str(y),np.argmin(mtc,axis=1))
      else:
        mtcCVCE = np.max(mtc,axis=1)
        mtcCVCE_epcs = reduce(lambda x,y: str(x)+'x'+str(y),np.argmax(mtc,axis=1))
      tmp[col] = np.mean(mtcCVCE)
      tmp[col+'_std'] = np.std(mtcCVCE)
      tmp[col+'_min'] = np.min(mtcCVCE)
      tmp[col+'_max'] = np.max(mtcCVCE)
      tmp[col+'_epcs'] = mtcCVCE_epcs
    if not Path(resDir+'/bestsCVCrossEpoch.csv').exists():
        bestCVCrossEpoch = pd.DataFrame(tmp,index=[0])
    else:
        if bestCVCrossEpoch is None: bestCVCrossEpoch = pd.read_csv(resDir+f'/bestsCVCrossEpoch.csv')
        bestCVCrossEpoch = bestCVCrossEpoch.append(tmp, ignore_index=True)
    bestCVCrossEpoch.to_csv(resDir+f'/bestsCVCrossEpoch.csv',index=False)
    res[full_name_comb] = [dfRes] # --> dropping individual kfold resutls
  #===================================================
  # GENERAL
  res[full_name_comb] = res[full_name_comb][0]
  tmp = {'model': full_name_comb}
  tmp.update({col : np.max(res[full_name_comb][col]) \
                for col in metrics \
                  if col not in MIN_METRICS})
  tmp.update({col : np.min(res[full_name_comb][col]) \
              for col in MIN_METRICS if col in res[full_name_comb].columns} )
  tmp.update({col+'_epc' : np.argmax(res[full_name_comb][col]) \
                for col in metrics \
                  if col not in MIN_METRICS})
  tmp.update({col+'_epc' : np.argmin(res[full_name_comb][col]) \
              for col in MIN_METRICS if col in res[full_name_comb].columns} )
  #===================================================
  # CV SPECIFIC
  if cv_K is not None:
    # CV specific
    for cvStat in ['_std','_min','_max']:
      tmp.update({col+cvStat : res[full_name_comb].loc[tmp[col+'_epc']][col+cvStat] \
                    for col in metrics})
  
  if not Path(resDir+'/bests.csv').exists():
    best = pd.DataFrame(tmp, index=[0])
  else:
    if best is None: best = pd.read_csv(resDir+f'/bests.csv')
    best = best.append(tmp, ignore_index=True)
  best.to_csv(resDir+f'/bests.csv',index=False)
  res[full_name_comb].to_csv(resDir+f'/training_{full_name_comb}.csv',index=True)
  return res, best, bestCVCrossEpoch