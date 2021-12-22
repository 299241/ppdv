import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import redis
import numpy as np
import json
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
log = app.logger

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)

def rms(x):
    return np.sqrt(sum(x**2/len(x)))


def quantile(x, q):
    return np.quantile(x, q)

def cos(inp):
    return inp

def full_df():
    data = db.xrevrange(1, count=50)
    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])

    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        df = df.append(pd.json_normalize(data_json['trace']['sensors']))

    for i in range(2, 7):
        data = db.xrevrange(i, count=50)
        for j in range(len(data)):
            data_json = json.loads(data[j][1]['data'])
            tmp = pd.json_normalize(data_json['trace']['sensors'])
            df = df.append(tmp)
    df = df.groupby(['name']).agg({
        'value': [
            'count', 'min', 'max', 'mean', ("rms", rms), 
            ("first quartile", lambda x: quantile(x, 0.25)), 
            ("median", lambda x: quantile(x, 0.5)), ("third quartile", lambda x: quantile(x, 0.75))
        ]
    })
    df.columns = [t[1] for t in df.columns.values]
    df.insert(loc=0, column='name', value=['L0', 'L1', 'L2', 'R0', 'R1', 'R2'])
    return df

def get_patient_table_data(id):
    data = db.xrevrange(id, count=10)

    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        df = df.append(pd.json_normalize(data_json['trace']['sensors']))

    cols = ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
    new_df = pd.DataFrame(columns=cols)
    for col in cols:
        sensor_data = df[df['name'] == col][['value', 'anomaly']]
        for i in range(sensor_data.shape[0]):
            if sensor_data.iloc[i]['anomaly'] != False:
                sensor_data.iloc[i]['value'] = '!' + str(sensor_data.iloc[i]['value'])
        sensor_data = sensor_data['value'].tolist()
        new_df[col] = sensor_data
    return new_df

def get_patient_table_start(val):
    table_data = get_patient_table_data(val)
    return table_data

app.layout = html.Div(children=[
    dcc.Interval(
        id='my-interval',
        interval=2000,
        disabled=False
    ),
    html.H1("All patients aggregated statistical data:"),
    dash_table.DataTable(
        id='full-table',
        columns=[{'name': i, "id": i} for i in full_df().columns]
    ),
    dcc.Tabs(id="patients-tabs", value='1', children=[
        dcc.Tab(label='Patient One', value='1'),
        dcc.Tab(label='Patient Two', value='2'),
        dcc.Tab(label='Patient Three', value='3'),
        dcc.Tab(label='Patient Four', value='4'),
        dcc.Tab(label='Patient Five', value='5'),
        dcc.Tab(label='Patient Six', value='6'),
    ]),
    html.Div(id='chosen-id', children=[
        html.Div(id='patient-name'),
        dbc.Card([
            dbc.CardBody(
                dash_table.DataTable(
                    id='patient-table',
                    columns=[{'name': i, "id": i} for i in get_patient_table_start("1").columns],
                    data=get_patient_table_start("1").to_dict('records')
                )
            ),
        ]),
        html.Div(children=[
            dbc.Card([
                dcc.Tabs(id="sensors-tabs", value='L0', children=
                [
                    dcc.Tab(label='L0', value='L0'),
                    dcc.Tab(label='L1', value='L1'),
                    dcc.Tab(label='L2', value='L2'),
                    dcc.Tab(label='R0', value='R0'),
                    dcc.Tab(label='R1', value='R1'),
                    dcc.Tab(label='R2', value='R2'),
                ]),
                daq.Gauge(
                    id='sensor-value',
                    label="Sensor Value:",
                    value=0,
                    min=0,
                    max=1023
                ),
                dbc.CardBody(
                    html.H1('wykresik punktowy')
                ),
            ])
        ])
        
        # wykresik punktowy z samymi anomaliami
        # wykresik liniowy z możliwością wyboru, które chcemy narysować jednocześnie na jednym wykresie i ograniczeniem czasu (RangeSlider)
    ])
])


@app.callback(
    Output('patient-table', 'data'),
    [Input('patients-tabs', 'value'),
    Input('my-interval', 'n_intervals')]
)
def change_tab(val, _):
    table_data = get_patient_table_data(val)
    return table_data.to_dict('records')


@app.callback(
    Output('full-table', 'data'),
    Input('my-interval', 'n_intervals')
)
def update_table(_):
    return full_df().to_dict('records')

@app.callback(
    [Output('sensor-value', 'value'),
    Output('sensor-value', 'label')],
    [Input('my-interval', 'n_intervals'),
    Input('sensors-tabs', 'value')],
    State('patients-tabs', 'value')
)
def update_gauge(_, sensor_name, patient_id):
    data = db.xrevrange(patient_id, count=1)
    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        df = df.append(pd.json_normalize(data_json['trace']['sensors']))
    df = df[df['name'] == sensor_name]
    return df.iloc[0]['value'], f'Sensor Value: {df.iloc[0]["value"]}'

@app.callback(
    Output('patient-name', 'children'), 
    Input('patients-tabs', 'value')
)
def update_name(id):
    data = db.xrevrange(id, count=1)
    data_json = json.loads(data[0][1]['data'])
    return [html.H2(f'{data_json["firstname"]} {data_json["lastname"]} ({data_json["birthdate"]})')]

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')
