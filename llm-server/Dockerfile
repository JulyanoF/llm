FROM python:3.9

WORKDIR /app

RUN pip install -r requirements.txt

COPY server.py /app/

CMD ["python", "server.py"]