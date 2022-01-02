FROM python:3.7

WORKDIR /dataVisualizer

COPY ./dataVisualizer /dataVisualizer

RUN pip install -r requirements.txt

EXPOSE 8050

CMD ["python3", "app.py"]