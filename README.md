

# ⚡ Energy Consumption Prediction System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
**CS 643: Cloud Computing – Programming Assignment 2**

> A scalable system for predicting energy consumption using Apache Spark on AWS EMR and deploying a containerized prediction service with Docker.

📄 [**Complete Walkthrough (PDF)**](https://github.com/KDShetty11/Energy-Consumption-Prediction-System-in-AWS-Cloud/blob/main/ks2378_Walkthrough_EnergyConsumptionPrediction.pdf)
📦 [**Docker Hub Image**](https://hub.docker.com/repository/docker/kdshetty/energypred)
📁 [**GitHub Repository**](https://github.com/KDShetty11/Energy-Consumption-Prediction-System-in-AWS-Cloud)

---

## 🧠 Project Overview

This project focuses on building a scalable energy consumption prediction pipeline using Apache Spark, AWS EMR, and Docker. The machine learning model is trained using Gradient Boosted Trees (GBT), optimized for RMSE (Root Mean Squared Error) evaluation.

---

## 🚀 Features

* **Distributed Model Training**: Spark-based training on a 4-node AWS EMR cluster.
* **Prediction Application**: Model served via Python on a standalone EC2 instance.
* **Containerized Deployment**: Dockerized prediction service for portability and ease of deployment.
* **Cloud-Optimized**: Utilizes EMR for parallel computation, EC2 for serving predictions, and HDFS for storage.

---

## 📁 Repository Structure

```
.
├── TrainingDataset.csv            # Training data
├── ValidationDataset.csv          # Validation data
├── train_model.py                 # Spark ML code to train GBT model
├── predict.py                     # Prediction script using trained model
├── Dockerfile                     # Docker image setup for prediction app
├── bootstrap.sh                   # EMR bootstrap script for dependencies
├── regressionmod/                 # Directory with trained GBT model (despite name)
└── model.tar.gz                   # Compressed trained model for deployment
```

---

## ⚙️ Setup & Usage

### 1. 🧪 Model Training on AWS EMR

* **Launch EMR Cluster**:

  * Version: `emr-7.8.0`
  * Spark: `3.5.5`
  * Instance type: 4 × `m5.xlarge`

* **Setup with Bootstrap**:

  ```bash
  ./bootstrap.sh  # Installs pandas and numpy
  ```

* **Upload Files**:

  * Transfer `TrainingDataset.csv` and `train_model.py` to EMR master via SFTP
  * Copy to HDFS:

    ```bash
    hadoop fs -put TrainingDataset.csv /user/hadoop/
    ```

* **Train Model**:

  ```bash
  spark-submit train_model.py
  ```

* **Download Model**:

  ```bash
  hadoop fs -get <model_folder> && tar -czvf model.tar.gz <model_folder>
  ```

---

### 2. 🔮 Prediction on EC2

* **Launch EC2 Instance**:

  * Type: `t2.medium`
  * OS: Ubuntu 24.04 LTS
  * Install: Python 3.8+, Java 8+, Spark 3.5.5

* **Transfer Files**:

  * Upload `predict.py`, `ValidationDataset.csv`, and `model.tar.gz`
  * Extract model:

    ```bash
    tar -xzvf model.tar.gz
    ```

* **Run Prediction**:

  ```bash
  spark-submit predict.py ValidationDataset.csv
  ```

---

### 3. 🐳 Docker Deployment

* **Build Image**:

  ```bash
  sudo docker build -t energypred .
  ```

* **Run Prediction in Container**:

  ```bash
  sudo docker run -v /home/ubuntu/ValidationDataset.csv:/app/ValidationDataset.csv energypred /app/ValidationDataset.csv
  ```

* **Or Pull from Docker Hub**:

  ```bash
  docker pull kdshetty/energypred
  ```

---

## ✅ Prerequisites

* AWS Account with EC2 and EMR access
* Python 3.8+
* Docker
* Apache Spark 3.5.5
* SSH and SFTP tools (e.g., OpenSSH, PuTTY, WinSCP)

---

## 📈 Evaluation Metric

* **Model Type**: Gradient Boosted Trees (GBT)
* **Performance Metric**: RMSE (Root Mean Squared Error)

> 🔍 *Note: Folder names may reference "regression," but the model is GBT (classification/regression ensemble).*



## 🙏 Acknowledgments

Developed for **CS 643: Cloud Computing**
Thanks to NJIT faculty and AWS documentation for guidance.

---

