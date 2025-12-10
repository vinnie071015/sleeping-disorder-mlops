# 💤 Sleep Disorder Prediction MLOps Pipeline

## 📖 Project Overview

This project implements a **production-grade MLOps pipeline** designed to predict sleep disorders (Insomnia, Sleep Apnea) based on lifestyle and health metrics.

It demonstrates a full lifecycle implementation—from data ingestion and cloud-based hyperparameter tuning to model deployment—using **AWS SageMaker**, **FastAPI**, **Streamlit**, and **Docker**. The system is architected to be "Production Ready," featuring automated CI/CD workflows and scalable containerized deployment.

-----

## 🏗️ Architecture & Tech Stack

This project leverages a modern cloud-native stack:

  * **☁️ Cloud Infrastructure:** AWS (SageMaker, S3, ECR, EC2)
  * **⚙️ Orchestration:** AWS SageMaker Training Jobs & Hyperparameter Tuner
  * **🧠 Models:** SVM (Champion), Random Forest, Logistic Regression
  * **🚀 Backend:** FastAPI (High-performance Inference Engine)
  * **🎨 Frontend:** Streamlit (Interactive User Interface)
  * **📦 Containerization:** Docker
  * **🔄 CI/CD:** GitHub Actions (Automated Build $\rightarrow$ Push to ECR $\rightarrow$ Deploy to EC2)

-----

## 📂 Repository Structure

```text
.
├── .github/workflows/   # CI/CD pipelines (deploy-model.yml)
├── api/                 # FastAPI backend application
│   ├── app.py           # API entry point
│   └── request_schema.json
├── docker/              # Docker configuration
│   └── Dockerfile       # Multi-stage build for App & UI
├── frontend/            # Streamlit dashboard
│   └── ui.py            # User interface code
├── notebooks/           # Jupyter Notebooks for experimentation
│   └── 01_sagemaker_orchestration.ipynb  # Main entry for SageMaker training
├── scripts/             # Utility scripts (e.g., S3 data upload)
├── src/                 # Core machine learning source code
│   ├── data_processor.py
│   └── train.py         # Training entry point used by SageMaker
├── tests/               # Unit and Smoke tests
├── config.yaml          # Project configuration
├── requirements.txt     # Python dependencies
└── run_services.sh      # Startup script for Docker container
-----
```

## 🚀 Getting Started

### 1\. Prerequisites (Tools)

Ensure you have the following installed:

  * **Python 3.9+**
  * **Docker Desktop** (Running)
  * **AWS CLI** (Configured with `aws configure`)

### 2\. Configuration (Secrets)

Before running any scripts, export the required environment variables in your terminal:

**Mac/Linux:**

```bash
# Required for Experiment Tracking
export WANDB_API_KEY="your_wandb_api_key"

# Required if running locally (outside SageMaker instances)
export AWS_ACCESS_KEY_ID="your_aws_key_id"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

**Windows (PowerShell):**

```powershell
$env:WANDB_API_KEY="your_wandb_api_key"
$env:AWS_ACCESS_KEY_ID="your_aws_key_id"
# ... and so on
```

### 3\. Local Installation (Non-Docker)

To run the application directly on your machine:

```bash
# 1. Clone the repository
git clone <repository_url>
cd sleeping-disorder-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the services (Backend + Frontend)
# Note: Ensure you have 'chmod +x run_services.sh' permissions
./run_services.sh
```

  * **Frontend:** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
  * **Backend API Docs:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)

### 4\. Docker Deployment (Recommended)

To replicate the production environment locally:

```bash
# 1. Build the image
docker build -t sleep-app:latest -f docker/Dockerfile .

# 2. Run the container
docker run -p 8000:8000 -p 8501:8501 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION \
  sleep-app:latest
```

-----

## 🧠 Model Training & Reproduction (SageMaker)

Follow these steps to reproduce the training results exactly as presented in the final report:
* Data comes from : https://www.kaggle.com/code/viniciusdiasx/predicting-sleep-disorders-with-machine-learning

1.  **Upload Raw Data:**
    Ensure `data/Sleep_health_and_lifestyle_dataset.csv` is present, then run:

    ```bash
    python scripts/upload_raw_data.py
    ```

    *This establishes the "Source of Truth" data in S3.*

2.  **Run Orchestration Notebook:**
    Open and run `notebooks/01_sagemaker_orchestration.ipynb`.

      * **Action:** Triggers the **HyperparameterTuner** on AWS SageMaker.
      * **Scope:** Trains three models in parallel (SVM, Random Forest, Logistic Regression).
      * **Result:** Automatically selects the best model based on Validation Accuracy.

3.  **Artifact Generation:**
    The champion model (`model.joblib` + `label_encoder.joblib`) is automatically packaged into `model.tar.gz` and saved to the S3 bucket defined in your config.

-----

## 🔌 API Documentation

The model is served via a **FastAPI** endpoint that adheres to SageMaker invocation standards.

**Endpoint:** `POST /invocations`

**Example Request:**

```json
{
  "gender": "Male",
  "age": 32,
  "occupation": "Software Engineer",
  "sleep_duration": 7.5,
  "quality_of_sleep": 8,
  "physical_activity_level": 60,
  "stress_level": 5,
  "bmi_category": "Normal",
  "blood_pressure": "120/80",
  "heart_rate": 70,
  "daily_steps": 4000
}
```

**Example Response:**

```json
{
    "prediction": "Insomnia"
}
```

*Possible values: `Missing`, `Insomnia`, `Sleep Apnea`*

-----

## 🔄 CI/CD Pipeline

The project utilizes **GitHub Actions** (`.github/workflows/deploy-model.yml`) for a fully automated deployment pipeline:

1.  **Trigger:** Push to `main` branch.
2.  **Build:** Creates a Docker image containing the latest code and model artifacts.
3.  **Push:** Uploads the image to **Amazon Elastic Container Registry (ECR)**.
4.  **Deploy:** Connects to the **EC2 instance** via SSH, pulls the new image, and seamlessly restarts the application container.