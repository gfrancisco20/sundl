"""
SXR-classes thresholds constants
"""

import numpy as np

__all__ = ['mpfTresh',
           'totehTresh'
          ]


""" Standard SXR-MPF based flare classes definition """
mpfTresh = {'quiet':(0.0,1e-7),
                'B':(1e-7,1e-6),
                'C':(1e-6,1e-5),
                'M':(1e-5,1e-4),
                'X':(1e-4,np.inf)}

""" SXR-TOTEH classes thresholds that replicate SXR-MPF 
climatology for different time-windows (dictionary keys in hours)
"""
totehTresh = {}

totehTresh[2] = {'quiet':(0.0    , 4e-5),
                     'B':(4e-5   , 1.8e-3),
                     'C':(1.8e-3 , 1.3e-2),
                     'M':(1.3e-2 , 1e-1),
                     'X':(1e-1, np.inf)}

totehTresh[4] = {'quiet':(0.0    , 3e-5),
                     'B':(3e-5   , 1.4e-3),
                     'C':(1.4e-3 , 9e-3),
                     'M':(9e-3 , 5e-2),
                     'X':(5e-2, np.inf)}

totehTresh[8] = {'quiet':(0.0    , 1.82e-5),
                     'B':(1.82e-5, 1.1e-3),
                     'C':(1.1e-3 , 7e-3 ),
                     'M':(7e-3  , 3.4e-2),
                     'X':(3.4e-2, np.inf)}

totehTresh[12] = {'quiet':(0.0    , 1.5e-5),
                     'B':(1.5e-5, 8.5e-4),
                     'C':(8.5e-4 , 6e-3 ),
                     'M':(6e-3  , 2.7e-2),
                     'X':(2.7e-2, np.inf)}

totehTresh[16] = {'quiet':(0.0    , 1e-5),
                     'B':(1e-5, 7e-4),
                     'C':(7e-4 , 5.5e-3 ),
                     'M':(5.5e-3  , 2.3e-2),
                     'X':(2.3e-2, np.inf)}

totehTresh[24] = {'quiet':(0.0    , 9e-6),
                     'B':(9e-6, 5e-4),
                     'C':(5e-4 , 4.6e-3 ),
                     'M':(4.6e-3  , 1.9e-2),
                     'X':(1.9e-2, np.inf)}

totehTresh[36] = {'quiet':(0.0    , 9e-6),
                     'B':(9e-6, 4e-4),
                     'C':(4e-4 , 3.8e-3 ),
                     'M':(3.8e-3  , 1.55e-2),
                     'X':(1.55e-2, np.inf)}

totehTresh[48] = {'quiet':(0.0    , 7e-6),
                     'B':(7e-6, 3.2e-4),
                     'C':(3.2e-4 , 3.4e-3 ),
                     'M':(3.4e-3  , 1.3e-2),
                     'X':(1.3e-2, np.inf)}

totehTresh[72] = {'quiet':(0.0    , 6e-6),
                     'B':(6e-6, 2.2e-4),
                     'C':(2.2e-4 , 2.7e-3 ),
                     'M':(2.7e-3  , 1e-2),
                     'X':(1e-2, np.inf)}

totehTresh[144] = {'quiet':(0.0    , 5e-6),
                     'B':(5e-6, 1.2e-4),
                     'C':(1.2e-4 , 1.7e-3 ),
                     'M':(1.7e-3  , 7.4e-3),
                     'X':(7.4e-3, np.inf)}