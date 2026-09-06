# Explainable Machine Learning for Forecasting Diabetic Complications Using NHS Data

## Live Dashboard

The deployed Streamlit dashboard is available here:

https://diabetes-ckd-forecasting.streamlit.app/

## About this project

This repository contains the practical work for my MSc Data Science dissertation.

The project investigates whether publicly available NHS diabetes care indicators can be used to forecast future chronic kidney disease rates across NHS Integrated Care Boards in England.

The analysis was completed in Python using Google Colab. The main notebook contains the complete workflow, including data preparation, geographical harmonisation, feature preparation, temporal validation, model comparison, final testing and model interpretation.

## Research focus

The main aim of the project was to investigate whether diabetes care information from one period could help predict chronic kidney disease outcomes in a later period.

The analysis focused on 42 NHS Integrated Care Boards and used historical National Diabetes Audit information together with chronic kidney disease outcome data.

Older NHS data used Clinical Commissioning Group geography, while newer data used Integrated Care Boards. Historical Clinical Commissioning Groups were therefore mapped into the current 42 Integrated Care Board areas using GP practice codes.

This allowed the same geographical areas to be compared consistently across different years.

## Data used

The project uses publicly available aggregated NHS data.

The main sources are:

National Diabetes Audit data from NHS England

Chronic kidney disease open data

Quality and Outcomes Framework data

The analysis does not use individual patient records or personally identifiable information.

The raw NHS data files are not included in this repository because of their size. They can be downloaded from the relevant NHS public data sources and uploaded when running the Google Colab notebook.

## Final modelling dataset

After cleaning and harmonising the data, the final forecasting panel contained:

42 Integrated Care Boards

4 predictor years from 2018/19 to 2021/22

4 chronic kidney disease target years from 2020 to 2023

168 Integrated Care Board year observations

19 harmonised National Diabetes Audit predictors

No missing predictor values in the final modelling panel

The forecasting structure was:

2018/19 indicators predict 2020 chronic kidney disease

2019/20 indicators predict 2021 chronic kidney disease

2020/21 indicators predict 2022 chronic kidney disease

2021/22 indicators predict 2023 chronic kidney disease

## Model development

Several forecasting approaches were tested, including:

Linear Regression

Ridge Regression

Random Forest

Gradient Boosting

A simple previous year chronic kidney disease baseline was also evaluated.

The models were assessed using chronological validation rather than randomly mixing observations from different years.

The final 2023 data was kept separate during model development and was only used after the final model had been selected.

## Final selected model

The final machine learning model was Ridge Regression with an alpha value of 1.0.

The model used previous year chronic kidney disease together with five compact National Diabetes Audit predictors:

Urine albumin monitoring

Serum creatinine monitoring

HbA1c target achievement

Blood pressure target achievement

Combined statin prevention

These variables were selected to provide a smaller and more clinically meaningful set of indicators while reducing the overlap between highly related measures.

## Final 2023 performance

The final Ridge model was evaluated on 42 unseen Integrated Care Boards from 2023.

The results were:

MAE: 7.446

RMSE: 9.117

R²: 0.786

This means the model reproduced much of the variation in chronic kidney disease rates across Integrated Care Boards in the final year.

However, the simple previous year chronic kidney disease baseline performed even better:

MAE: 5.838

RMSE: 7.385

R²: 0.860

This was an important finding of the study. It suggests that chronic kidney disease rates at Integrated Care Board level show strong year to year persistence and that the additional diabetes care indicators did not improve short term forecasting accuracy beyond the simple previous year baseline.

## Model interpretation

The Ridge coefficient analysis showed that previous year chronic kidney disease was by far the strongest predictor in the final model.

The remaining diabetes care indicators made smaller contributions to the predictions.

These relationships should be interpreted as model associations rather than evidence that any individual indicator directly causes chronic kidney disease to increase or decrease.

## How to run the project

1. Open the Google Colab notebook included in this repository.

2. Run the notebook cells in order.

3. When prompted, upload the required NHS data files.

4. The notebook will clean and harmonise the data automatically.

5. Continue running the cells to reproduce the forecasting dataset, model comparison, final 2023 evaluation and model interpretation.

## Main notebook

`Diabetic_Complications_XAI_Dissertation.ipynb`

The notebook contains the complete reproducible analysis used for this project.

## Final conclusion

The study shows that explainable machine learning can successfully model regional chronic kidney disease patterns using aggregated public NHS data.

The final Ridge model achieved strong performance on the unseen 2023 data, but it did not outperform the simpler previous year chronic kidney disease baseline.

The most important finding was therefore not that machine learning automatically improved forecasting, but that previous chronic kidney disease levels contained extremely strong information about future regional outcomes.

This also highlights the importance of using realistic baselines and chronological validation when evaluating forecasting models.
