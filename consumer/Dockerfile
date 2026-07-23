FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY consumer.py .
COPY signal_model.pkl .

CMD ["python", "consumer.py"]