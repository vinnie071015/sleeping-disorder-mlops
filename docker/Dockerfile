# 1. Base Image
FROM --platform=linux/amd64 python:3.9-slim

# 2. Working Directory
WORKDIR /app

# 3. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Dependency Installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Core Code
COPY api/ api/
COPY src/ src/
# --- New: Copy Frontend Code ---
COPY frontend/ frontend/ 

# 6. Copy Model Files
COPY notebooks/best_model_extracted/ notebooks/best_model_extracted/

# 7. --- New: Copy and set up the startup script ---
COPY run_services.sh .
RUN chmod +x run_services.sh

# 8. --- Modification: Expose two ports ---
# 8000 is for the API, 8501 is for the Streamlit page
EXPOSE 8000 8501

# 9. --- Modification: Change the startup command to run the script ---
CMD ["./run_services.sh"]