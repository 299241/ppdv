FROM python:3.7
WORKDIR /dataManager

ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 8080
ENV FLASK_ENV development

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT flask run --with-threads --reload