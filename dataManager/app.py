from flask import Flask
import redis
import requests
import json
import time

app = Flask(__name__)
log = app.logger

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)


@app.route('/')
def main():
    while True:
        for id in range(1, 7):
            getData(id)
        time.sleep(2)


@app.route('/xd')
def xd():
    return "<html>Ala</html>"


def getData(id):
    resp = requests.get(f'http://tesla.iem.pw.edu.pl:9080/v2/monitor/{id}')
    db.xadd(id, {'data': resp.text})
    # log.info(f"stream len = {db.xlen(id)}")
    # if db.xlen(id) > 120:
    #     # log.info("obcinam")
    #     db.xtrim(id, 60)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', threaded=True)
