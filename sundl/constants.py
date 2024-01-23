

MIN_METRICS = ['far',
              'loss', 
              'MAE', 
              'RMSE', 
              'MAE_dec', 
              'RMSE_dec', 
              'MAE_tf', 
              'RMSE_tf']
MIN_METRICS = MIN_METRICS + ['val_'+mtc for mtc in MIN_METRICS]