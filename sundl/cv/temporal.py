"""
Utilities to build independant CV folds in temporal/time-series context
"""

import datetime
import numpy as np
import pandas as pd

from sundl.utils.flare import class_stats, flux2cls
from sundl.utils.flare.thresholds import mpfTresh

__all__ = ['sampleFromFolds',
           'instantiateFolds',
           'instantiateFoldsSequentially',
           'chunks2fold_balancedAllocation',
           'buildChunks'
           ]

def buildChunks(samplesDates,
                clsTresholds = None,
                buffer = pd.DateOffset(days=27),
                chunk_width = pd.DateOffset(days=81),
                colIdCls = 'cls',
                ):
  df = samplesDates
  if clsTresholds is not None:
    classes = clsTresholds.keys()
    df=class_stats(df, classes = classes, colIdCls = colIdCls, verbose=1)
  else:
    classes = []

  chunks = None
  # chunks = pd.DataFrame(dict(**{'chunkId':[], 'start':[], 'end':[]},**{cls:[] for cls in classes}))
  chunksId = 0
  initChunk = True
  for idx in range(len(df)):
    shift = chunksId*buffer
    date = df.index[idx] + shift
    if date<df.index.max():
      if initChunk:
        if chunks is None:
          chunks =  pd.DataFrame(dict(**{'chunkId':chunksId, 'start':date, 'end':None},**{cls:0 for cls in classes}),index=[chunksId])
        else:
          chunks = pd.concat([chunks,
                              pd.DataFrame(dict(**{'chunkId':chunksId, 'start':date, 'end':None},**{cls:0 for cls in classes}),index=[chunksId])
                              ])
        initChunk = False
      if date < chunks.loc[chunksId,'start'] + chunk_width:
        for cls in classes:
          chunks.loc[chunksId,cls] += df.loc[date,cls]
      else:
        chunks.loc[chunksId,'end'] = date
        chunksId+=1
        initChunk = True
  chunks.loc[chunksId,'end'] = df.index.max()
  return chunks

def chunks2fold_balancedAllocation(chunks, 
    n_fold, 
    clsBalanceImportance = {'X':4, 'M':2, 'C':1, 'B':1, 'quiet':4}
):
  # CONSTANTS
  f = 1
  min_num_per_fold = {'X' :int(chunks['X'].sum()*f / n_fold),
                      'M' :int(chunks['M'].sum()*f / n_fold),
                      'C' :int(chunks['C'].sum()*f / n_fold),
                      'B' :int(chunks['B'].sum()*f / n_fold),
                      'quiet' :int(chunks['quiet'].sum()*f / n_fold),
                      }
  importanceBalanceCorrected = False
  if importanceBalanceCorrected:
    weights = chunks.sum()
    # print(weights)
    clsBalanceImportance = {cls : clsBalanceImportance[cls] / weights[cls] for cls in clsBalanceImportance.keys()}
    # print(clsBalanceImportance)
  classes = list(clsBalanceImportance.keys())
  folds = pd.DataFrame({'foldId':range(n_fold), 'chunks':[[] for k in range(n_fold)], 'X': np.zeros(n_fold), 'M': np.zeros(n_fold),'C': np.zeros(n_fold),'B': np.zeros(n_fold),'quiet': np.zeros(n_fold)},index=range(n_fold))
  usedChunk = []

  ################################################################################
  # init
  availableChunks = chunks[~chunks['chunkId'].isin(usedChunk)].reset_index(drop=True)
  balances = np.ones(n_fold) * np.inf
  while len(availableChunks)>0:
    # possibilities = {idxChunk:balances for idxChunk in availableChunks.index}
    foldsMin = balances.copy()
    foldsMinChunkId = np.ones(n_fold) * availableChunks.index.values[0]
    foldsClsCounts = {k:{cls : folds.loc[k,cls] for cls in classes} for k in range(n_fold)}
    foldsClsExcess = {k: {cls : folds.loc[k,cls] >= min_num_per_fold[cls] for cls in classes} for k in range(n_fold)}
    for k in range(n_fold):
      for idxChunk in availableChunks.index:
        new_clsCounts = {cls : folds.loc[k,cls] + availableChunks.loc[idxChunk, cls] for cls in classes}
        new_balance = np.sum([clsBalanceImportance[cls]*np.abs(min_num_per_fold[cls]-new_clsCounts[cls])/min_num_per_fold[cls] for cls in classes])
        # possibilities[idxChunk, k] = new_balance
        # print(k, new_balance)
        if new_balance < foldsMin[k]:
          # print('update : ', k, new_balance, foldsMin[k])
          foldsMinChunkId[k] = idxChunk
          foldsClsCounts[k] = new_clsCounts
          foldsClsExcess[k] = {cls : foldsClsCounts[k][cls] >= min_num_per_fold[cls] for cls in classes}
          foldsMin[k] = new_balance
    k = 0
    diff = balances[k] - foldsMin[k]
    for kprime in range(n_fold):
      diffp = balances[kprime] - foldsMin[kprime]
      if diffp > diff:
        k = kprime
        diff = diffp
    balances[k] = foldsMin[k]
    idxChunk = foldsMinChunkId[k]
    chunkId = availableChunks.loc[idxChunk, 'chunkId']
    folds.loc[k,'chunks'].append(chunkId)
    for cls in classes:
      folds.loc[k,cls] += availableChunks.loc[idxChunk, cls]
    usedChunk.append(chunkId)
    availableChunks = chunks[~chunks['chunkId'].isin(usedChunk)].reset_index(drop=True)
  print('chunks : ', availableChunks)
  return folds

def instantiateFolds(fl_history,
                     chunks, 
                     folds, 
                     testSplitDate = None, 
                     bufferTest = pd.DateOffset(days=27),  
                     classes = mpfTresh.keys(),
                     verbose = 1
  ):
  # Train / test split
  if testSplitDate is not None:
    dfTest = fl_history[fl_history.index >= testSplitDate + bufferTest]

    dfTrain = fl_history[fl_history.index < testSplitDate]
  else:
    dfTrain = fl_history
    dfTest = None

  dfFolds = {}

  for k in folds.index:
    foldChunks = folds.loc[k,'chunks']
    if isinstance(foldChunks,str):
      foldChunks = foldChunks.strip('[').strip(']').split(',')
    for i,chunkId in enumerate([int(float(chunkId)) for chunkId in foldChunks]):
      if i==0:
        dfFolds[k] = dfTrain[(dfTrain.index>chunks.loc[chunkId,'start']) & (dfTrain.index<chunks.loc[chunkId,'end'])]
      else:
        dfFolds[k] = pd.concat([dfFolds[k],
                                dfTrain[(dfTrain.index>chunks.loc[chunkId,'start']) & (dfTrain.index<chunks.loc[chunkId,'end'])]
                                ],axis=0)
    dfFolds[k] = dfFolds[k].sort_index()
    df = dfFolds[k]
    df['cls'] =  df['mpf'].apply(lambda x: flux2cls(x))
    print(f'\nFOLD {k}:')
    _=class_stats(df, classes = classes, colIdCls = 'cls', verbose=verbose)
  return dfFolds, dfTest

def sampleFromFolds(dfFolds,
                    balance     = True,
                    foldSize    = 2000, # for unbalanced case
                    clsBinsSize = {'quiet' : 400,
                                   'B'     : 400,
                                   'C'     : 400,
                                   'M'     : 400,
                                   'X'     : 400},
                    verbose = 1
                    ):
  classes = list(clsBinsSize.keys())
  dfFoldsBalanced ={}
  for k in dfFolds.keys():
    if balance:
      dfCls = {}
      for cls in ['quiet','B','C','M', 'X']:
        if clsBinsSize[cls] == 0:
          continue
        num_cls = len( dfFolds[k][dfFolds[k]['cls'] == cls])
        if num_cls > clsBinsSize[cls]:
          thinFactor = int(num_cls/clsBinsSize[cls])
          if thinFactor>0:
            dfCls[cls] = dfFolds[k][dfFolds[k]['cls'] == cls][::thinFactor]
          else:
            dfCls[cls] = dfFolds[k][dfFolds[k]['cls'] == cls]
          # The undersampling is done prefeerably by increasing the time resolution; then we complete with random undersampling
          dfCls[cls] = dfCls[cls].sample(clsBinsSize[cls])
        else:
          dfCls[cls] =  dfFolds[k][dfFolds[k]['cls'] == cls]
        # print(h, cls, len(dfCls[cls]))

      dfFoldsBalanced[k] = pd.concat(list(dfCls.values()),axis=0).sort_index()
    else:
      thinFactor = int(len(dfFolds[k])/foldSize)
      dfFoldsBalanced[k] = dfFolds[k][::thinFactor]
      dfFoldsBalanced[k] = dfFolds[k].sample(foldSize)

    print(f'\nFOLD {k}: {len(dfFoldsBalanced[k])}')
    _=class_stats(dfFoldsBalanced[k], classes = classes, colIdCls = 'cls', verbose=verbose)
  return dfFoldsBalanced

def instantiateFoldsSequentially(
    chunks,
    timeserie,
    n_folds = 5,
    testType  = 'folds', #param [ 'folds', 'temporal' ] {type:"string"}
    testDate  = datetime.datetime(2019,1,1,0,0,0), # for temporal test split only
    excludeDateRanges = None
):
  folds = {idx:None for idx in range(n_folds)}
  if testType == 'temporal':
    dfTest = timeserie[timeserie.index >= testDate + pd.DateOffset(days=27)]
    timeserie = timeserie[timeserie.index < testDate]
  if excludeDateRanges is not None:
    for startExclude, endExclude in excludeDateRanges:
      timeserie = timeserie[(timeserie.index <= startExclude) | (timeserie.index > endExclude)]
      

  # we build n_folds by filling them sequentially with available chunks
  for chunkIdx in range(len(chunks)):
    foldIdx = chunkIdx % n_folds
    sampleIds = (timeserie.index >= chunks.loc[chunkIdx,'start']) & (timeserie.index < chunks.loc[chunkIdx,'end'])
    if folds[foldIdx] is None:
      folds[foldIdx] = timeserie[sampleIds]
    else:
      folds[foldIdx] = pd.concat([folds[foldIdx], timeserie[sampleIds]], axis = 0)

  # we prepare the train,val,test folds combinations
  readyFolds = []
  for k in folds.keys():
    dfVal = folds[k]
    trainIdxs = list(folds.keys())
    trainIdxs.remove(k)
    if testType == 'folds':
      dfTest = folds[(k+1) % n_folds]
      trainIdxs.remove((k+1) % n_folds)
    for i,trainIdx in enumerate(trainIdxs):
      if i==0:
        dfTrain = folds[trainIdx]
      else:
        dfTrain = pd.concat([dfTrain,folds[trainIdx]],axis=0)

    if testType == 'folds':
      readyFolds.append([dfTrain,dfVal,dfTest])
    else:
      readyFolds.append([dfTrain,dfVal])
  if testType == 'folds':
    return readyFolds
  else:
    return readyFolds, dfTest