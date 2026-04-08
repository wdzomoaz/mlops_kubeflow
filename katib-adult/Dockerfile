FROM python:3.11
RUN pip install pandas==2.0.3 numpy==1.24.3 scikit-learn==1.3.0
COPY train.py /train.py
ENTRYPOINT ["python", "/train.py"]
