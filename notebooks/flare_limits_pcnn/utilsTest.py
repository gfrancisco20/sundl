from config import * # path to sundl

import datetime
import numpy as np
import pandas as pd

from sundl.utils.data import read_Dataframe_With_Dates
from sundl.utils.flare import SCs

def initPerfTest(typePerf = 'fd'):
  """
  typePerf : str, @['fd', 'patches', 'patchesTot']
  """
  perfTest = pd.DataFrame(
    {'model':[],
    'modelFdId':[],
    'startingDate':[],
    'filter':[],
    'thd':[],
    'tss':[],
    'hss':[],
    'mcc':[],
    'f1':[],
    'fss':[],
    'recall':[],
    'precision':[],
    'far':[],
    'acc_w':[],
    'acc':[],
    'tp':[],
    'tn':[],
    'fp':[],
    'fn':[],
    'p':[],
    'n':[],
    'c':[],
    'tot':[],
    'bal_pos':[],
    'switch_rate':[]
    }
  )
  if typePerf == 'patches':
    perfTest['limb'] = []
    perfTest['patchId'] = []
  elif typePerf == 'patchesTot':
    perfTest['group'] = []
  return perfTest
  
def filterCollection(dfRes, ptcTag = ''):
  filters = {}

  filters['all'] = np.ones(len(dfRes),dtype='bool')
  filters['SC_25_Peak'] = dfRes.index >= datetime.datetime(2022,1,1)


  # pathTestSet = F_PATH_TEST(labelCol,h)
  # dfFilter= read_Dataframe_With_Dates(pathTestSet)
  # dfFilter.index = dfFilter.index + pd.DateOffset(hours= -h)
  
  # filters['peak24_rates'] = dfRes.index.isin(dfFilter.index)
  filters['peak24_rates'] = (dfRes.index >= SCs['peak24'][0]) & (dfRes.index <= SCs['peak24'][1])

  filters['windowChanging'] = dfRes[f'change{ptcTag}']
  filters['windowConstant'] = ~filters['windowChanging']
  
  return filters

def fullDiskPerformance(fdPredictions, startingDates, filterNames, includeFolds, pThds = [0.5]):
  perfTest = initPerfTest()
  for startingDate in startingDates:
    for filterName in filterNames: # startDates filterNames
      for modelName in fdPredictions.keys(): # ['M+_mpf_Persistant','M+_mpf_PTx8xRtdAllxEasyposX_A2e5x25epc_blos_24',]
        dfRes = fdPredictions[modelName].copy()

        filters = filterCollection(dfRes.copy())
        filterDate =  dfRes.index >= startingDate 

        filter = filters[filterName] & filterDate
        
        predCols = [col for col in dfRes.columns if col[:3]=='pre']
        foldIds = [col.split('_')[1] for col in predCols if col[-1]!='d']
        for thd in pThds:#np.arange(0,1,0.01):

          if includeFolds:
            predTypes = ['avg'] + foldIds
          else:
            predTypes = ['avg']

          for modelFdId in predTypes:
            if modelFdId == 'avg':
              predCol = 'pred'
            else:
              predCol = f'pred_{modelFdId}'
            skip = False
            if len(perfTest) > 0:
              if len(perfTest[(perfTest['model']==modelName) \
                              & (perfTest['filter']==filterName) \
                              & (perfTest['thd']==thd) \
                              & (perfTest['modelFdId']==modelFdId)
                              ])>0:
                skip = True

            if not skip:
              # filter = filters[filterName]
              # filter = filtersDates[date]
              
              dfResFiltered = dfRes.copy()[filter]
              predFilterd = dfResFiltered[predCol]>=thd

              tp = ((predFilterd == 1) & (dfResFiltered['label']==1)).sum()
              tn = ((predFilterd == 0) & (dfResFiltered['label']==0)).sum()
              fp = ((predFilterd == 1) & (dfResFiltered['label']==0)).sum()
              fn = ((predFilterd == 0) & (dfResFiltered['label']==1)).sum()
              c = dfResFiltered[f'change'].sum()
              p = tp + fn
              n = tn + fp
              tot = p+n
              tss = tp / (tp+fn) + tn / (tn+fp) - 1
              hss = 2*(tp*tn - fn*fp) / (p*(fn+tn) + n*(tp+fp))
              # nmcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(p)*(n)*(tn+fn))
              mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

              prec = tp / (tp+fp)
              far = 1-prec
              rec = tp / p
              f1 = 2*prec*rec / (prec+rec)

              f_pers = (p - c/2) / p
              fssp = f1 - f_pers
              fssp = fssp / np.abs((1+np.sign(fssp))/2-f_pers)

              f_rand = 2*p / (3*p+n)
              fssr = f1 - f_rand
              fssr = fssr / np.abs((1+np.sign(fssr))/2-f_rand)

              acc = (tp + tn) / (p+n)
              acc_w = (tp/p + tn/n) / (tp/p + fn/p + tn/n + fp/n)
              bal_pos = p/tot
              switch_rate = dfResFiltered['change'].sum() / len(dfResFiltered['change'])
              # switch_rate = dfResFiltered[dfResFiltered['label']==True]['change'].sum() / p
              perfTest = pd.concat([perfTest,pd.DataFrame({
                'model':modelName,
                'modelFdId':modelFdId,
                'startingDate':startingDate, # filterName , date
                'filter':filterName, # ''
                'thd':thd,
                'tss':tss,
                'hss':hss,
                'mcc':mcc,
                'f1':f1,
                'fssp':fssp,
                'fssr':fssr,
                'recall':rec,
                'precision':prec,
                'far':far,
                'acc_w':acc_w,
                'acc':acc,
                'tp':tp,
                'tn':tn,
                'fp':fp,
                'fn':fn,
                'p':p,
                'n':n,
                'c':c,
                'tot':tot,
                'bal_pos':bal_pos,
                'switch_rate':switch_rate},index=[len(perfTest)+1])], axis=0)
  return perfTest

def patchesPerformance(ptPredictions, startingDates, filterNames, includeFolds, pThds = [0.5]):
  perfTestPtTot = initPerfTest('patchesTot')
  perfTestByPatch = initPerfTest('patches')
  for startingDate in startingDates:
    for filterName in filterNames:
      for modelName in ptPredictions.keys():

        dfRes = ptPredictions[modelName].copy()
        

        ptcPredCols = [col for col in ptPredictions[modelName].columns if len(col)==len('pred_pt0') and col[:3]=='pre']
        ptcPredColsFolds = [col for col in ptPredictions[modelName].columns if len(col)==len('pred_pt0_fd001') and col[:3]=='pre']
        foldIds = list(set([col.split('_')[2] for col in ptcPredColsFolds if col[-1]!='d']))




        for thd in pThds:#np.arange(0,1,0.01):

          if includeFolds:
            predTypes = ['avg'] + foldIds
          else:
            predTypes = ['avg']

          for modelFdId in predTypes:

            skip = False
            if len(perfTestPtTot) > 0:
              if len(perfTestPtTot[(perfTestPtTot['model']==modelName) \
                              & (perfTestPtTot['filter']==filterName) \
                              & (perfTestPtTot['thd']==thd) \
                              & (perfTestPtTot['modelFdId']==modelFdId)
                              ])>0:
                skip = True

            if not skip:



              for patchId in range(len(ptcPredCols)):
                filters = filterCollection(dfRes.copy(), ptcTag = f'_pt{patchId}')
                filterDate =  dfRes.index >= startingDate 
                filter = filters[filterName] & filterDate
                dfResFiltered = dfRes[filter].copy()
                
                if modelFdId == 'avg':
                  predCol = f'pred_pt{patchId}'
                else:
                  predCol = f'pred_pt{patchId}_{modelFdId}'

                predFilterd = dfResFiltered[predCol]>=thd

                # predCols = [col for col in dfRes.columns if col[:3]=='pre']
                # predCols = [col for col in predCols if int(col.split('_')[1][-1])==patchId]

                tp = ((predFilterd == 1) & (dfResFiltered[f'label_pt{patchId}']==1)).sum()
                tn = ((predFilterd == 0) & (dfResFiltered[f'label_pt{patchId}']==0)).sum()
                fp = ((predFilterd == 1) & (dfResFiltered[f'label_pt{patchId}']==0)).sum()
                fn = ((predFilterd == 0) & (dfResFiltered[f'label_pt{patchId}']==1)).sum()
                c = dfResFiltered[f'change_pt{patchId}'].sum()
                p = tp + fn
                n = tn + fp
                tot = p+n
                tss = tp / (tp+fn) + tn / (tn+fp) - 1
                hss = 2*(tp*tn - fn*fp) / (p*(fn+tn) + n*(tp+fp))
                mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
                prec = tp / (tp+fp)
                far = 1-prec
                rec = tp / p
                f1 = 2*prec*rec / (prec+rec)

                f_pers = (p - c/2) / p
                fssp = f1 - f_pers
                fssp = fssp / np.abs((1+np.sign(fssp))/2-f_pers)

                f_rand = 2*p / (3*p+n)
                fssr = f1 - f_rand
                fssr = fssr / np.abs((1+np.sign(fssr))/2-f_rand)

                acc = (tp + tn) / (p+n)
                acc_w = (tp/p + tn/n) / (tp/p + fn/p + tn/n + fp/n)
                bal_pos = p/tot

                if patchId in [0,3,4,7]:
                  limb = True
                else:
                  limb = False

                switch_rate = dfResFiltered[f'change_pt{patchId}'].sum() / len(dfResFiltered)
                perfTestByPatch = pd.concat([perfTestByPatch,pd.DataFrame({'model':modelName,
                                                            'modelFdId':modelFdId,
                                                            'patchId':patchId,
                                                            'limb':limb,
                                                            'startingDate':startingDate,
                                                            'filter':filterName,
                                                            'thd':thd,
                                                            'tss':tss,
                                                            'hss':hss,
                                                            'mcc':mcc,
                                                            'f1':f1,
                                                            'fssp' : fssp,
                                                            'fssr' : fssr,
                                                            'recall':rec,
                                                            'precision':prec,
                                                            'far':far,
                                                            'acc_w':acc_w,
                                                            'acc':acc,
                                                            'tp':tp,
                                                            'tn':tn,
                                                            'fp':fp,
                                                            'fn':fn,
                                                            'p':p,
                                                            'n':n,
                                                            'tot':tot,
                                                            'bal_pos':bal_pos,
                                                            'switch_rate':switch_rate
                                                            },index=[len(perfTestByPatch)+1])], axis=0)
              for group in ['all','limb','center']:
                tmp = perfTestByPatch[((perfTestByPatch.model == modelName) \
                                    & (perfTestByPatch.modelFdId == modelFdId) \
                                    & (perfTestByPatch['filter'] == filterName) \
                                    & (perfTestByPatch.thd == thd) \
                                    )].copy()
                if group == 'limb':
                  tmp = tmp[tmp['limb'] == True]
                elif group == 'center':
                  tmp = tmp[tmp['limb'] == False]
                tp = tmp['tp'].sum()
                tn = tmp['tn'].sum()
                fp = tmp['fp'].sum()
                fn = tmp['fn'].sum()
                c = (tmp['switch_rate']*tmp['tot']).sum()
                switch_rate = c / tmp['tot'].sum() #tmp['switch_rate'].mean()
                p = tp + fn
                n = tn + fp
                tot = p+n
                tss = tp / (tp+fn) + tn / (tn+fp) - 1
                hss = 2*(tp*tn - fn*fp) / (p*(fn+tn) + n*(tp+fp))
                prec = tp / (tp+fp)
                far = 1-prec
                rec = tp / p
                f1 = 2*prec*rec / (prec+rec)

                f_pers = (p - c/2) / p
                fssp = f1 - f_pers
                fssp = fssp / np.abs((1+np.sign(fssp))/2-f_pers)

                f_rand = 2*p / (3*p+n)
                fssr = f1 - f_rand
                fssr = fssr / np.abs((1+np.sign(fssr))/2-f_rand)

                acc = (tp + tn) / (p+n)
                acc_w = (tp/p + tn/n) / (tp/p + fn/p + tn/n + fp/n)
                bal_pos = p/tot
                perfTestPtTot = pd.concat([perfTestPtTot,pd.DataFrame({'model':modelName,
                                                              'modelFdId':modelFdId,
                                                              'group':group,
                                                              'startingDate':startingDate,
                                                              'filter':filterName,
                                                              'thd':thd,
                                                              'tss':tss,
                                                              'hss':hss,
                                                              'mcc':mcc,
                                                              'f1':f1,
                                                              'fssp' : fssp,
                                                              'fssr' : fssr,
                                                              'recall':rec,
                                                              'precision':prec,
                                                              'far':far,
                                                              'acc_w':acc_w,
                                                              'acc':acc,
                                                              'tp':tp,
                                                              'tn':tn,
                                                              'fp':fp,
                                                              'fn':fn,
                                                              'p':p,
                                                              'n':n,
                                                              'tot':tot,
                                                              'bal_pos':bal_pos,
                                                              'switch_rate':switch_rate},index=[len(perfTestPtTot)+1])], axis=0)
  return perfTestPtTot, perfTestByPatch