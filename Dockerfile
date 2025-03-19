FROM python:3.10.12

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
COPY ./main.py /app/main.py

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
