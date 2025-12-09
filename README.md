import os
## 📖 Project Overview
This project implements a production-grade MLOps pipeline to predict sleep disorders (Insomnia, Sleep Apnea) based on lifestyle and health metrics. It demonstrates a full lifecycle from data ingestion to model deployment using **AWS SageMaker**, **FastAPI**, **Streamlit**, and **Docker**.

The system is designed to be "Production Ready," featuring automated CI/CD workflows, containerized deployment, and cloud-based hyperparameter tuning.

## 🏗️ Architecture & Tech Stack

* **Cloud Infrastructure:** AWS (SageMaker, S3, ECR, EC2)
* **Orchestration:** AWS SageMaker Training Jobs & Hyperparameter Tuner
* **Model:** SVM (Champion), Random Forest, Logistic Regression
* **Backend:** FastAPI (Inference Engine)
* **Frontend:** Streamlit (User Interface)
* **Containerization:** Docker
* **CI/CD:** GitHub Actions (Build -> Push to ECR -> Deploy to EC2)

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
```
# 🚀 Getting Started

1.  **Prerequisites**
    *   Python 3.9+
    *   Docker Desktop installed
    *   AWS CLI configured with appropriate permissions (S3, SageMaker, ECR)

2.  **Local Installation & Testing**
    To run the application locally without Docker:

    ```bash
    # Clone the repository
    git clone <repository_url>
    cd sleeping-disorder-mlops

    # Install dependencies
    pip install -r requirements.txt

    # Run the services (Backend + Frontend)
    chmod +x run_services.sh
    ./run_services.sh
    ```
    Frontend: Access at http://localhost:8501

    Backend API: Access at http://localhost:8000/docs

3.  **Docker Deployment (Recommended)**
    To replicate the production environment locally:

    ```bash
    # Build the image
    docker build -t sleep-app:latest -f docker/Dockerfile .

    # Run the container
    docker run -p 8000:8000 -p 8501:8501 sleep-app:latest
    ```

# 🧠 Model Training & Reproduction (SageMaker)

To reproduce the model training results exactly as presented in the report:

*   **Upload Raw Data:** Ensure `data/Sleep_health_and_lifestyle_dataset.csv` exists locally, then run:

    ```bash
    python scripts/upload_raw_data.py
    ```
    This uploads the "Source of Truth" data to S3.

*   **Run Orchestration Notebook:** Open `notebooks/01_sagemaker_orchestration.ipynb`.

    This notebook triggers the HyperparameterTuner on AWS SageMaker.

    It trains three models in parallel: SVM, Random Forest, and Logistic Regression.

    It automatically selects the best model based on Accuracy.

*   **Artifacts:** The best model (model.joblib and label_encoder.joblib) will be saved to the S3 bucket specified in `config.yaml` and packaged into `model.tar.gz`.

# 🔌 API Documentation

The model is served via a FastAPI endpoint compatible with SageMaker invocation standards.

*   **Endpoint:** `POST /invocations`

*   **Request Body Example:**

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
*   **Response:**

    ```json
    {
        "prediction": "None"
    }
    ```
    (Possible values: "None", "Insomnia", "Sleep Apnea")

# 🔄 CI/CD Pipeline

The project utilizes GitHub Actions (`.github/workflows/deploy-model.yml`) for automated deployment:

*   **Push to Main:** Triggers the workflow.
*   **Build:** Builds the Docker image containing the code and model artifacts.
*   **Push to ECR:** Uploads the image to Amazon Elastic Container Registry.
*   **Deploy to EC2:** SSHs into the EC2 instance, pulls the new image, and restarts the container.