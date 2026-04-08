FROM python:3.11-slim
WORKDIR /app
RUN pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0 minio==7.2.0
COPY train.py .
ENTRYPOINT ["python", "train.py"]
