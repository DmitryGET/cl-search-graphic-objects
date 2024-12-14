FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN mkdir -p /app/static /app/uploads

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
