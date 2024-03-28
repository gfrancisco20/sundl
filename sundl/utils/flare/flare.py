"""
Flare general function utilies
"""

import datetime
import pandas as pd

from sundl.utils.flare.thresholds import mpfTresh, totehTresh
from sundl.utils.flare.windows import windowHistoryFromFlList

__all__ = ["flux2cls", 
           "class_stats",
           "SCs",
           "climatologicalRates",
           
           ]


def flux2cls(flux, detailed=False, clsRangeDict=mpfTresh):
  """
  Convert a continuous value 'flux' into categorical ones
  following the classes given in 'clsRangeDict'

  Parameters
  ----------
  flux : float
    continuous value to be categorized
  detailed : bool, optional
    default is false, if 'true' add {flux:.1e} at the end of the final class
    e.g. returns X2 instead of X in the case of flares' SXR-classes
  clsRangeDict : dict, optional
    keys are classes/categories in which to onvert flux,
    and values their corresponding ranges,
    default to 'mpfTresh' -> convert 'flux' in SXR-MPF classes
  """
  try:
    for clsTest in clsRangeDict.keys():
      if flux >= clsRangeDict[clsTest][0] and flux < clsRangeDict[clsTest][1]:
        cls = clsTest
    if detailed and (not list(clsRangeDict.keys()).index(cls) == 0):
      cls = cls + "{:.1e}".format(flux)[:3]
    cls = cls  # to generate an exception if no cls is found
  except Exception as e:
    print(e)
    print(flux)
  return cls


def class_stats(
  df,
  classes=["quiet", "B", "C", "M", "X"],
  colIdCls="cls",
  clsTransfo=None,
  verbose=0,
):
  """
  Add boolean columns for each element in 'classes'
  with value equal to 1 if the column 'colIdCls' is equal to the corresponding element

  Parameters
  ----------
  df : DataFrame
    dataframe on which to add classes boolean columns
  classes : List, optional
    list of classes, default to flares SXR-MPF classes
  colIdCls : str, optional
    df column that give the class/category of each sample, default to 'cls'
  clsTransfo : func, optional
    eventual transform applied to colIdCls to return the wanted classes,
    default to None,
    e.g. with colIdCls='mpf' use clsTransfo = flux2cls
  verbose : int, optional
    if > 0 returns classes statistics, default to 0
  """
  df = df.copy()
  df = df.sort_values(by=[colIdCls], ignore_index=False, inplace=False)
  if clsTransfo is not None:
    df[colIdCls] = df[colIdCls].apply(lambda cls: clsTransfo(cls))
  for cls in classes:
    df[cls] = df[colIdCls].apply(lambda x: 1 if x == cls else 0)
  if verbose > 0:
    for cls in classes:
      num = df[cls].sum()
      pct = 100 * num / len(df)
      print(f"{cls}-flares samples: {num} ({pct:.2f}%)")
    df[classes].describe()
  df = df.sort_index()
  return df

start_SC_22 = datetime.datetime(1986,1,1)
end_SC_22 = datetime.datetime(1996,8,1)
end_SC_23 = datetime.datetime(2008,12,1)
end_SC_24 = datetime.datetime(2019,12,1)
start_SC_25_Asc = datetime.datetime(2020,1,1)
end_SC_25_Asc = datetime.datetime(2023,5,1)

SCs = {'all':(start_SC_22,end_SC_24), # 3 SC average
       '22': (start_SC_22,end_SC_22), 
       '23': (end_SC_22,end_SC_23), 
       '24': (end_SC_23,end_SC_24),
       '24_sdo' : (datetime.datetime(2010,5,1),end_SC_24),
       'peak24': (datetime.datetime(2010,12,1), datetime.datetime(2018,12,31)),
       'SC_25_Asc': (start_SC_25_Asc, end_SC_25_Asc)
       }

def climatologicalRates(flCatalog, 
                        listSizesTimeWindowsH,
                        colPeakFlux = 'peak_flux',
                        colIntFlux = 'flux_integral',
                        colClasses = 'cat',
                        classes = [],
                        timeRes_h = 2
                        ):
  """
  Compute climatological rates for time-windows of sizes listSizesTimeWindowsH
  from flare event catalog flCatalog
  """
  climato_rates = {}
  for window_h in listSizesTimeWindowsH:
    climato_rates[window_h] = {'sc': [], 'class': [], 'cr_mpf':[], 'cr_toteh': []}
    fl_history = windowHistoryFromFlList(flCatalog, 
                                         window_h = window_h, 
                                         timeRes_h = timeRes_h, 
                                         minDate = None, 
                                         maxDate = None,
                                         colPeakFlux = colPeakFlux,
                                         colIntFlux = colIntFlux,
                                         colClasses = colClasses,
                                         classes = classes
                                         )
    for cls in mpfTresh.keys():
      for sc in SCs.keys():
        start = SCs[sc][0]
        end = SCs[sc][1]
        df = fl_history.copy()
        if start is not None:
          df = df[df.index > start]
        if end is not None:
          df = df[df.index < end]
        n = len(df)
        mpf   = len(df[(df['mpf']   >= mpfTresh[cls][0] )            & (df['mpf'] < mpfTresh[cls][1] )]) / n * 100
        toteh = len(df[(df['toteh'] >= totehTresh[window_h][cls][0] )& (df['toteh'] < totehTresh[window_h][cls][1] )]) / n  * 100
        climato_rates[window_h]['sc'].append(sc)
        climato_rates[window_h]['class'].append(cls)
        climato_rates[window_h]['cr_mpf'].append(mpf)
        climato_rates[window_h]['cr_toteh'].append(toteh)
    climato_rates[window_h] = pd.DataFrame(climato_rates[window_h])
  return climato_rates
