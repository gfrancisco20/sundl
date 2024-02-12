

MIN_METRICS = ['far',
              'loss', 
              'MAE', 'mae',
              'MAEP', 'maep',
              'RMSE', 'rmse',
              'MAE_dec', 'mae_dec',
              'RMSE_dec', 'rmse_dec',
              'MAE_tf', 'mae_tf',
              'RMSE_tf', 'rmse_tf'
              ]
MIN_METRICS = MIN_METRICS + ['val_'+mtc for mtc in MIN_METRICS]