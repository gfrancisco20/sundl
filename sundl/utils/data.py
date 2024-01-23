"""
Data and filesystem utilies
"""

from glob import glob
from pathlib import Path
import datetime
import numpy as np
import pandas as pd


# from sundl.utils.colab import PATH_IMAGES

__all__ = ['loadMinMaxDates',
           'read_Dataframe_With_Dates'
           ]

def loadMinMaxDates(pathFolder,
                    folderStruct = '*/*/*/*',
                    minOffsetH   = 24,
                    maxOffseyD   = 1
                    ):
  """
  This function walk the folder 'pathFolder' organized with the
  structure 'wavelength/YYYY/MM/DD' to return the smallest and biggest
  dates of the folder
  
  Parameters
  ----------
  pathFolder : PosixPath / str, optional
    folder containing files organized with a date based structre,
    default to PATH_IMAGES from sundl.utils.colab
  folderStruct : str, otpional
    structure of pathFolder, default to */*/*/* 
    where the date structure starts only after the first child folder
    e.g. works for a structure 'wavelength/YYYY/MM/DD'
    and will return the minDate and maxDate among all wavelength subfolders
  minOffsetH : int, optional
    offset in hours to subsrtact to the minimum date, default to 24
  maxOffsetD : int, optional
    offset in days to add to the maximum date
  """

  if isinstance(pathFolder,str):
    pathFolder = Path(pathFolder)
  dates = glob((pathFolder/folderStruct).as_posix())
  dates = [date for date in dates if date.split('/')[-3].isnumeric()]
  dates = np.array([datetime.datetime(int(date.split('/')[-3]),int(date.split('/')[-2]),int(date.split('/')[-1])) for date in dates])
  
  minDate = dates.min() + pd.offsets.DateOffset(hours=-minOffsetH)
  maxDate = dates.max() + pd.offsets.DateOffset(days=maxOffseyD)
  
  return minDate, maxDate
  
def read_Dataframe_With_Dates(pathCsvDf, tsColumns = ['timestamp'], tsFmt = '%Y-%m-%d %H:%M:%S', colAsIndex = 'timestamp'):
  df = pd.read_csv(pathCsvDf)
  for tsColumn in tsColumns:
    if tsColumn in df.columns:
      df[tsColumn] = df[tsColumn].apply(lambda x: datetime.datetime.strptime(x,tsFmt)) # '%Y/%m/%d/H%H00/
      if tsColumn == colAsIndex:
        df = df.set_index([tsColumn], drop = True)
  return df