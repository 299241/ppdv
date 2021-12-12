from flask import Flask
import redis
import pandas as pd
import numpy as np
import json

app = Flask(__name__)
log = app.logger

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)


def rms(x):
    return np.sqrt(sum(x**2/len(x)))


def quantile(x, q):
    return np.quantile(x, q)


@app.route('/')
def main():
    data = db.xrevrange(1, count=60)
    log.info(len(data))
    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])

    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        df = df.append(pd.json_normalize(data_json['trace']['sensors']))

    for i in range(2, 7):
        data = db.xrevrange(i, count=60)
        log.info(f"{i}: {len(data)}")
        for j in range(len(data)):
            data_json = json.loads(data[j][1]['data'])
            tmp = pd.json_normalize(data_json['trace']['sensors'])
            df = df.append(tmp)
    df = df.groupby(['name']).agg({'value':
                                   ['count', 'min', 'max', 'mean', ("rms", rms), ("xxxd", lambda x: quantile(x, 0.25)), (lambda x: quantile(x, 0.5)), (lambda x: quantile(x, 0.75))]})
    # df = df.reset_index()
    return df.to_html()


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
