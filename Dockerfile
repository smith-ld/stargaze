FROM python:3.12
LABEL authors="Lucas Smith"

RUN pip install numpy keras tensorflow

CMD ["python3", "main.py"]