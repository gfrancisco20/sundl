"""
Functions to build time-windows labels and features from event catalogs
"""

import datetime
import numpy as np
import pandas as pd

__all__ = ['windowHistoryFromFlList',
           'windowHistoryFromFlList_ByPatchSector_EXACT'
           ]

def windowHistoryFromFlList(flCatalog, 
                            window_h, 
                            timeRes_h, 
                            minDate = None, 
                            maxDate = None,
                            colPeakFlux = 'peak_flux',
                            colIntFlux = 'flux_integral',
                            colClasses = 'cat',
                            classes = ['A','B','C','M','X']
                            ):
  """
  This function compute mpf, rate, tote and toteh for time-windows of 'window_h' from a catalog of flare event flCatalog
  The results is given as a  feature/history of a given time-window, i.e. at date D, the values characterize the interval 
  [D - window_h ; D[
  To obtain labels for the time-windows, i.e. value characterizing [D ; D + window_h [ at date D, 
  one can simply shift the index of the output : fl_labels = fl_history.shift(periods =  -window_h, freq='H')
  
  Parameters
  ----------
  flCatalog : DataFrame
    cataog of flare event
  window_h : int
    size of the time-windows (in hours) on which to compute the windows feature
  timeRes_h : int
    time resolution at which to compute the features
  colPeakFlux : str, optional
    column name of the peak_flux column in flCatalog, 
    requiered for mpf and rate computation,
    set to None to skip mpf and rate
    default to 'peak_flux'
  colIntFlux : str, optional
    column name of the flux_integral column in flCatalog, 
    required to compute tote and toteh,
    set to None to skip tote and toteh
    default to 'flux_integral'
  colClasses : str, optional
    column name of the class column in flCatalog,
    used to computee rates and totes at class levels
    default to 'cat
  classes : List(str), optionals
    class on which to compute tote and rate at class levels,
    default to ['A','B','C','M','X']
    
  Returns
  -------
  DataFrame
    a dataframe of (window_h)H-time-windows features at a resolution of (timeRes_h)H
  """
  flares = flCatalog.copy()
  # flares['timestamp'] = flares['timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')) # '%Y/%m/%d/H%H00/
  # flares = flares.set_index('timestamp',drop = True)
  freq = f'{timeRes_h}h'
  window =  f'{window_h}h'
  features = {}

  if colPeakFlux is not None:
    features['mpf']   = flares[colPeakFlux].resample(freq, label ='right', closed='right').max()
    features['rate'] = flares[colPeakFlux].resample(freq, label ='right', closed='right').count()
    for fclass in classes:
        features[f'rate_{fclass}'] = flares[flares[colClasses]==fclass][colPeakFlux].resample(freq, label ='right', closed='right').count()
  if colIntFlux is not None:
    features['tote'] = flares[colIntFlux].resample(freq, label ='right', closed='right').sum()
    for fclass in classes:
        features[f'tote_{fclass}'] = flares[flares[colClasses]==fclass][colIntFlux].resample(freq, label ='right', closed='right').sum()
        
  if window_h > timeRes_h:
    if colPeakFlux is not None:
      features['mpf']  = features['mpf'].rolling(window, closed = 'right').max()
      features['rate'] = features['rate'].rolling(window, closed = 'right').sum()
      for fclass in classes:
        features[f'rate_{fclass}'] = features[f'rate_{fclass}'].rolling(window, closed = 'right').sum()
    if colIntFlux is not None:
      features['tote'] = features['tote'].rolling(window, closed = 'right').sum()
      for fclass in classes:
        features[f'tote_{fclass}'] = features[f'tote_{fclass}'].rolling(window, closed = 'right').sum()
        
  if colIntFlux is not None:
    features['toteh'] = features['tote'] / window_h
      
  fl_history = pd.DataFrame(features)[2*int(window_h/timeRes_h):-2*int(window_h/timeRes_h)].fillna(0)
  if minDate is not None:
    fl_history = fl_history[(fl_history.index>minDate)]
  if maxDate is not None:
    fl_history = fl_history[(fl_history.index<maxDate)]
  return fl_history

def windowHistoryFromFlList_ByPatchSector_EXACT(
    flCatalog_positions,
    missing_positions_events,
    window_h,
    timeRes_h,
    minDate = None,
    maxDate = None,
    mpfId = 'singleID',
    t2f_as_label = True, # if False t2f computed as history (negative value)
    num_patches = 8,
    CMX = True
):
  '''
  Much slower than windowHistoryFromFlList_ByPatchSector but exact label sector attribution accounting for rotation
  '''
  timestamps = pd.date_range(start='2020/01/01', end='2023/05/01', freq=f'{timeRes_h}H')

  mpfTresh = {'A':(0.0,1e-7),
          'B':(1e-7,1e-6),
          'C':(1e-6,1e-5),
          'M':(1e-5,1e-4),
          'X':(1e-4,np.inf)}
  def initStructWindow():
    struct = {'timestamp':[],
            'mpf':[],
            'rate':[],
            'rate_A':[],
            'rate_B':[],
            'rate_C':[],
            'rate_M':[],
            'rate_X':[],
            'tote':[],
            'toteh':[],
            'tote_A':[],
            'tote_B':[],
            'tote_C':[],
            'tote_M':[],
            'tote_X':[],
              mpfId:[],
            't2mpf_h':[],
            'x':[],
            'y':[],
            'sector':[]}
    return struct
  def sector_8_ptc(x,y,t2f_h = None):
    im_size = 4096
    patch_size = im_size // 4
    row = y // patch_size

    # rotation correction
    if t2f_h is not None:
      if t2f_h == 0:
        return None

      ylim = 614 / 0.6
      Rpix = 959.63 / 0.6
      ord = y - im_size //2
      if ord > 0:
        ord = min(ord,ylim)
      else:
        ord = max(ord,-ylim)
      ref = 2 * np.sqrt(Rpix**2 - ord**2) # ref = 2*Rpix
      half_carr = 27 * 24 / 2
      rot =  ref * t2f_h / half_carr
      x = x - rot

    if x < 0:
      x=0

    col = x // patch_size


    # if row in [0,1]:
    #   row = 1
    # elif row in [2,3]:
    #   row = 0

    if row in [0,1]:
      row = 0
    elif row in [2,3]:
      row = 1


    sector = row*4 + col
    return int(sector)

  fl_historys = {}
  for sector in range(num_patches):
    fl_historys[sector] = pd.DataFrame(initStructWindow())

  ########################################################
  ########################################################
  for end in timestamps[(window_h//2):]:
    start = end - pd.offsets.DateOffset(hours = window_h)
    flares = flCatalog_positions[(flCatalog_positions.index > start) & (flCatalog_positions.index <= end)].copy()
    if len(flares)>0:
      flares['t2mpf_delay'] = flares['tstart'].apply(lambda x: x-end)
      flares[f't2mpf_h'] = flares['t2mpf_delay'].apply(lambda x: x.total_seconds()/3600)
      if t2f_as_label:
        flares[f't2mpf_h'] = flares[f't2mpf_h'].apply(lambda x: window_h + x)
      flares['sectors'] = flares[['x','y','t2mpf_h']].apply(lambda xyt: sector_8_ptc(xyt[0],xyt[1],xyt[2]),axis=1)#.fillna(int(ptchIdx))
      for sector in range(num_patches):
        struct = initStructWindow()
        flaresSector = flares[flares['sectors']==sector].copy()
        if len(flaresSector)>0:
          struct['timestamp'] = end
          struct['sector'] = sector
          struct['rate'] = len(flaresSector)
          # mpf
          idxMpf = flaresSector['peak_flux'].argmax()
          struct['mpf'] = flaresSector['peak_flux'].iloc[idxMpf]
          struct[mpfId] = flaresSector[mpfId].iloc[idxMpf]
          struct['x'] = flaresSector['x'].iloc[idxMpf]
          struct['y'] = flaresSector['y'].iloc[idxMpf]
          struct['t2mpf_h'] = flaresSector['t2mpf_h'].iloc[idxMpf]
          # tote
          struct['tote'] = flaresSector['flux_integral'].sum()
          struct['toteh'] = struct['tote'] / window_h
          # clss
          for cls in ['A','B','C','M','X']:
            flaresSectorClass = flaresSector[(flaresSector['peak_flux'] >= mpfTresh[cls][0]) & (flaresSector['peak_flux'] < mpfTresh[cls][1])].copy()
            struct[f'rate_{cls}'] = len(flaresSectorClass)
            struct[f'tote_{cls}'] = flaresSectorClass['flux_integral'].sum()
        else:
          struct = initStructWindow()
          struct['timestamp'] = end
          struct['sector'] = sector
          struct['mpf'] = 0
          struct['rate'] = 0
          struct['tote'] = 0
          struct['toteh'] = 0
          struct[mpfId] = -1
          struct['t2mpf_h'] = 24
          struct['x'] = -1
          struct['y'] = -1
          for cls in ['A','B','C','M','X']:
            struct[f'rate_{cls}'] = 0
            struct[f'tote_{cls}'] = 0
        fl_historys[sector] = pd.concat([fl_historys[sector],pd.DataFrame(struct,index=[len(fl_historys[sector])])], axis=0)
    else:
      for sector in range(num_patches):
        # quiet window
        struct = initStructWindow()
        struct['timestamp'] = end
        struct['mpf'] = 0
        struct['rate'] = 0
        struct['tote'] = 0
        struct['toteh'] = 0
        struct[mpfId] = -1
        struct['t2mpf_h'] = 24
        struct['x'] = -1
        struct['y'] = -1
        struct['sector'] = sector
        for cls in ['A','B','C','M','X']:
          struct[f'rate_{cls}'] = 0
          struct[f'tote_{cls}'] = 0
        fl_historys[sector] = pd.concat([fl_historys[sector],pd.DataFrame(struct,index=[len(fl_historys[sector])])], axis=0)
  for sector in range(num_patches):
    fl_historys[sector] = fl_historys[sector].set_index('timestamp')
    for end_exclusion in missing_positions_events:
      start_exclusion = end_exclusion + pd.DateOffset(hours= -window_h)
      # print(start_exclusion, end_exclusion)
      fl_historys[sector] = fl_historys[sector][(fl_historys[sector].index <= start_exclusion) | (fl_historys[sector].index >= end_exclusion)]
  return fl_historys