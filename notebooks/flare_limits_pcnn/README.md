# PCNN and Flare Forecasting Models Limitations

This folder shows how to reproduce the results of the paper :
"Limits of solar flare forecasting models and new deep learning approach"

More specifically notebooks are provided to :
- prepare independant training and validation folds for full-disk forecasts
- train Patch-Distributed-CNNs (PCNNs)
- derive regional risk-prediction and positions estimations from a trained PCNN
- highlight limits of flare forecasting models,
  mainly their lack of skill to forecast changes of activities and systematic underperformance compared to persistent models

## Notebooks

### [0_CV_Folds](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/0_CV_Folds.ipynb)
Prepare and store indepedant folds for cross-validation

### [1_Training](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/1_Training.ipynb)
CV-training on the folds resulting from the previous notebook

### [2_Model_Selection](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/2_Model_Selection.ipynb)
Analysis of the CV results and model selection

### [3_Test_Predictions](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/3_Test_Predictions.ipynb)
Computation and storage of the predictions on the operational tests set

### [4_Test_Evaluation](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/4_Test_Evaluation.ipynb)
Test performace analysis, including :
- full-disk level performances
- patch/sector level performances
- center vs. limb pathes performances
- performances on windows where the activity differes from the previous one (AC-windows)
- performances on windows where the activity is the same as the previous one (NC-windows)
- persistent relative metrics

### [5_Explainability](https://github.com/gfrancisco20/sundl/blob/master/notebooks/flare_limits_pcnn/5_Explainability.ipynb)
Visual explainability results of the models prediction and estimations of forecasted event positions

## Data

The SDO images used to train and evaluate the models are available from the [SDOCompactML Dataset](https://doi.org/10.5281/zenodo.10465437)    
Only small_blos.zip and small_0193x0211x0094.zip are used in this work (448x448 pixels images)    
Download both zip files and unzip them in PATH_IMAGES (c.f. Setup section or config.py file)  

The flare catalogs used to derived time-windows labels are available from the [Plutino Flare Catalog](https://doi.org/10.5281/zenodo.10560188)  
Download all files and place them in PATH_ROOT_DRIVE_DS/'Events_Catalogs' (c.f. Setup section config.py file)  

## Setup

The project is organised with the  structure provided below.  
Set your own PATH_ROOT_DRIVE and PATH_ROOT_LOCAL in config.py  
Then run config.initProject() to create the project folders.  
Place zipped images files in PATH_ROOT_DRIVE/Datasets/Images (or alternatively directly unzipped in PATH_IMAGES (PATH_ROOT_LOCAL/'images')  
Place event catalogs from [Plutino Flare Catalog](https://doi.org/10.5281/zenodo.10560188) in PATH_ROOT_DRIVE/'Datasets/Events_Catalogs'  
CV-results from the paper are already available in the project's folder 'Results_Paper_PCNN'  
Trained model and test prediction are not provided for storage reasons.  
If you want to rerun all results without retraining models and computing prediction (skipping notebook 1 and 3)   
feel free to ask me the saved trained models and prediction 
and respectively place them in PATH_RES/'Results_Paper_PCNN/models' and PATH_RES/'Results_Paper_PCNN/pred_test'
```
PATH_ROOT_DRIVE
-- Datasets (PATH_ROOT_DRIVE_DS)
   -- Images
      -> zipped image datasets
         dataset dowloadable from : https://doi.org/10.5281/zenodo.10465437 
   -- Events_Catalogs
      -> flare_catalog_file (PATH_FLCATALOG)
      -> flare_catalog_file_with_positions (PATH_FLCATALOG_WITH_POS)
      -> flare_with_unmatched_positions_dates (PATH_MISSING_POS_DATES)
         files dowloadable from : https://doi.org/10.5281/zenodo.10560188
   -- Time_Series
      -> windows_features_files (F_PATH_WINDOWS)
   -- Meta
      -> climatological_rates_file (PATH_CR_RATES)
-- Folds
   -- Flare
      -- Chunks (temporal_chunks_files F_PATH_CHUNKS)
      -- Allocation (maps_chunks_folds_alloc_files F_PATH_ALLOC)
      -- FoldsTrainVal (train_val_folds_files F_PATH_FOLDS)
      -- OperationalTest (test_set_files F_PATH_TEST)

PATH_ROOT_LOCAL
-- images (PATH_IMAGES)
   folder in which zipped files from 'PATH_ROOT_DRIVE_DS/Images' are unzipped


PATH_RES
-- Results_Paper_PCNN
   -> log.csv                  , models training status
   -> bests.csv                , best epoch-averages metrics
   -> bestsCVCrossEpoch.csv    , best cross-epoch metrics
   -> training_{modelName}.csv , fold-averages training and valiation curves
   -- training_folds
      -> training_{modelName}_fd{foldId}.csv , folds training and valiation curves
   -- models
      -- {modelName}            , saved trained model
      -> {modelName}_config.pkl , saved model config (metrics, optimizer, dataset parameters etc.)
   -- pred_test
      -> {modelName}_fd.csv            , full-disk predictions
      -> {modelName}_fd_withLabels.csv , full-disk predictions with labels
      -> {modelName}_pt.csv            , patches/sector predictions
      -> {modelName}_pf_withLabels.csv , patches/sector predictions with labels
```

