# Stroke Prediction Pipeline

## Overview

This repository contains a Python script that implements a data processing and machine learning pipeline to predict which patients will have a heart stroke. The pipeline includes data loading, preprocessing, feature engineering, model training, evaluation, and email notifications.

## Features

- Data loading from a CSV file
- Data cleaning and preprocessing
- Handling of missing values and outliers
- Standardizing numerical features
- Label encoding for categorical features
- RandomForestClassifier model for classification
- Evaluation metrics including accuracy, precision, recall, F1 score, and ROC AUC
- Email notifications for pipeline execution results or errors

## Requirements

- Python 3.x

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/moe94z/Heart_Stroke_Prediction.git
   cd Heart_Stroke_Prediction
2. Install necessary packages (pull requirements.txt) 
   ```bash
   pip install -r requirements.txt

  (contain all the necessary libraries)

## Execution
python3 Prod_Heart_Stroke_Prediction.py > errors.log &

## Airflow Dag script for Apache Airflow execution
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stroke_pipeline',
    default_args=default_args,
    description='A pipeline for stroke prediction',
    schedule_interval=timedelta(days=1),
)

t1 = BashOperator(
    task_id='run_pipeline',
    bash_command='python3 /local/environment/prod/stroke_prediction_pipeline.py > /local/environment/prod/errors.log',
    dag=dag,
)

t1







