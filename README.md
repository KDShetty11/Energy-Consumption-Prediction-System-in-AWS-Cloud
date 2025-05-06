Energy Consumption Prediction System

Overview

This repository contains the implementation of Programming Assignment 2 for CS 643: Cloud Computing. The project focuses on building an energy consumption prediction system using Apache Spark for parallel model training on an AWS EMR cluster, deploying a prediction application on a single EC2 instance, and containerizing the application using Docker. The model, based on Gradient Boosted Trees (not regression, despite folder naming), predicts energy consumption and evaluates performance using RMSE.


[Complete Walkthrough](https://github.com/KDShetty11/Energy-Consumption-Prediction-System-in-AWS-Cloud/blob/main/ks2378_Walkthrough_EnergyConsumptionPrediction.pdf)

Features:

Parallel Model Training: Utilizes Apache Spark on an AWS EMR cluster with four EC2 instances for distributed training.
Prediction Application: Runs on a single EC2 instance, loading the trained model to make predictions on validation datasets.
Docker Integration: Containerizes the prediction application for portability and deployment, hosted on Docker Hub.
AWS Integration: Leverages AWS EMR for training and EC2 for prediction, with HDFS for distributed storage.

Repository Structure :

-> TrainingDataset.csv: Dataset used for training the model.

-> ValidationDataset.csv: Dataset used for prediction and evaluation.

-> train_model.py: Spark script for training the Gradient Boosted Trees model.

-> predict.py: Script for loading the trained model and making predictions.

-> Dockerfile: Defines the Docker image for the prediction application.

-> bootstrap.sh: Script for installing dependencies on EMR cluster nodes.

-> regressionmod: Trained model(gradient boost)

Prerequisites:

AWS account with access to EMR and EC2 services.
Docker installed for building and running the container.
Python 3.8+ with dependencies: numpy, pandas, pyspark.
Apache Spark 3.5.5.
SSH client (e.g., OpenSSH, PuTTY) and SFTP tool (e.g., WinSCP).

Setup and Usage
1. Parallel Model Training on AWS EMR

Create an EMR Cluster:
Launch an EMR cluster (emr-7.8.0, Spark 3.5.5) with 4 m5.xlarge instances.
Use bootstrap.sh to install numpy and pandas.


Upload Files:
Transfer TrainingDataset.csv and train_model.py to the EMR master node via SFTP.
Copy files to HDFS: hadoop fs -put <file> /user/hadoop/.


Train the Model:
Run spark-submit train_model.py to train the model.
Download the trained model (model.tar.gz) from HDFS via SFTP.



2. Prediction on EC2 Instance

Launch EC2 Instance:
Create a t2.medium EC2 instance with Ubuntu 24.04 LTS.
Install dependencies: Python, Java, Spark 3.5.5.


Upload Files:
Transfer predict.py, ValidationDataset.csv, and model.tar.gz via SFTP.
Extract the model: tar -xzvf model.tar.gz.


Run Prediction:
Execute spark-submit predict.py ValidationDataset.csv to generate predictions and RMSE.



3. Docker Deployment

Build Docker Image:
Install Docker on the EC2 instance.
Build the image: sudo docker build -t energypred ..


Run Container:
Run the prediction: sudo docker run -v /home/ubuntu/ValidationDataset.csv:/app/ValidationDataset.csv energypred /app/ValidationDataset.csv.


Pull from Docker Hub:
Pull the pre-built image: docker pull kdshetty/energypred.



Docker Hub:

The Docker image is available at: kdshetty/energypred.

Notes:

Ensure sufficient storage and permissions for EMR and EC2 instances.
Validate predictions using ValidationDataset.csv to check RMSE.
Test the Docker container on a fresh EC2 instance for consistency.
Folder names may reference "regression," but the model is Gradient Boosted Trees.

AI Usage:

ChatGPT/Copilots: Used for initial train_model.py structure and partial Dockerfile. Adapted for dataset-specific needs and optimized for RMSE.
Custom Code: Parameter tuning in train_model.py and full predict.py implementation were written from scratch.

Links:

GitHub Repository: https://github.com/KDShetty11/Energy-Consumption-Prediction-System-in-AWS-Cloud

Docker Hub Repository: https://hub.docker.com/repository/docker/kdshetty/energypred

Acknowledgments:

This project was developed as part of CS 643: Cloud Computing, with guidance from course materials and AWS documentation.
