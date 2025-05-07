FROM python:3.12
LABEL authors="Lucas Smith"

RUN pip install numpy keras tensorflow spacy bs4 pandas
RUN pip install scikit-learn
RUN python -m spacy download en_core_web_md

COPY . .
CMD ["python3", "main.py"]