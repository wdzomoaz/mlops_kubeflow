FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
WORKDIR /app
RUN pip install torchvision==0.22.0 scikit-learn --no-cache-dir
COPY train.py .
ENTRYPOINT ["python", "train.py"]
