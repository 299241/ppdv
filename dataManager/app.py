import redis
import requests
import time

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)


def getData(id):
    resp = requests.get(f'http://tesla.iem.pw.edu.pl:9080/v2/monitor/{id}')
    db.xadd(id, {'data': resp.text})
    if db.xlen(id) > 300:
        db.xtrim(id, 300)


if __name__ == '__main__':
    print('Service started!')
    try:
        while True:
            for id in range(1, 7):
                getData(id)
            time.sleep(1.7)
    except KeyboardInterrupt:
        print('Service stopped!')
