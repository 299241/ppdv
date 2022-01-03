import dash
from dash import html, dcc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import redis
import numpy as np
import json
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import dash_daq as daq
from datetime import datetime, time, timedelta
from dash.exceptions import PreventUpdate


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
log = app.logger

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)


def rms(x):
    return np.sqrt(sum(x**2/len(x)))


def quantile(x, q):
    return np.quantile(x, q)


def get_patient_data(id):
    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    data = db.xrevrange(id, count=50)
    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        tmp = pd.json_normalize(data_json['trace']['sensors'])
        tmp['date'] = datetime.utcfromtimestamp(
            int(data[j][0].split('-')[0])//1000)
        df = df.append(tmp)
    df = df.reset_index(drop=True)
    return df


def get_statistical_data(df):
    df = df.groupby(['name']).agg({
        'value': [
            'count', 'min', 'max', 'mean', ("rms", rms),
            ("first quartile", lambda x: quantile(x, 0.25)),
            ("median", lambda x: quantile(x, 0.5)),
            ("third quartile", lambda x: quantile(x, 0.75))
        ]
    })
    df.columns = [t[1] for t in df.columns.values]
    df.insert(loc=0, column='name', value=['L0', 'L1', 'L2', 'R0', 'R1', 'R2'])
    return df


def get_patient_table_data(df):
    df = df.iloc[:60]
    cols = ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
    new_df = pd.DataFrame(columns=['date', *cols]) 

    for col in cols:
        sensor_data = df[df['name'] == col][['value', 'anomaly', 'date']] 
        for i in range(sensor_data.shape[0]):
            if sensor_data.iloc[i]['anomaly'] != False:
                sensor_data.iloc[i]['value'] = '!' + \
                    str(sensor_data.iloc[i]['value'])
        sensor_data = sensor_data['value'].tolist()
        new_df[col] = sensor_data
    dates = df[df['name'] == 'L0'][['date']]['date'].tolist()
    new_df['date'] = dates
    return new_df

app.layout = html.Div(children=[
    dcc.Interval(
        id='quick-interval',
        interval=2000,
        disabled=False
    ),
    dcc.Interval(
        id='slow-interval',
        interval=10000,
        disabled=False
    ),
    dcc.Store(id='cached-data'),
    dcc.Store(id='cached-patient-data'),
    html.H1("All patients aggregated statistical data:"),
    dash_table.DataTable(
        id='full-table',
        columns=[{'name': i, "id": i} for i in ['name', 'count', 'min', 'max', 'mean', 'rms', 'first quartile', 'median', 'third quartile']]
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
                    columns=[{'name': i, "id": i}
                             for i in ["date", "L0", "L1", "L2", "R0", "R1", "R2"]]
                )
            ),
        ]),
        html.Div(children=[
            dbc.Card([
                dcc.Tabs(id="sensors-tabs", value='L0', children=[
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
                html.Div(id='anomaly-graph')
            ])
        ]),
        html.Div([
            dcc.Dropdown(id='sensors-graph-filter', options=[
                {'value': x, 'label': x} for x in ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
            ], multi=True, value=['L0', 'R0']),
            dcc.RangeSlider(id='sensors-graph-time-filter', min=-10,
                            max=0, value=[-10, 0], tooltip={"placement": "bottom"}),
            dcc.Graph(id='sensors-graph')
        ])
    ])
])


@app.callback(
    Output('cached-data', 'data'),
    Input('quick-interval', 'n_intervals')
)
def cached_data(_):
    df = get_patient_data(1)
    for i in range(2, 7):
        df.append(get_patient_data(i))
    return df.to_json()


@app.callback(
    Output('patient-table', 'data'),
    [Input('patients-tabs', 'value'),
    Input('cached-patient-data', 'modified_timestamp')],
    State('cached-patient-data', 'data')
)
def change_tab(patient_id, ts, data):
    if ts is None:
        raise PreventUpdate
    return [] if not data else get_patient_table_data(pd.read_json(data)).to_dict('records')


@app.callback(
    Output('full-table', 'data'),
    Input('slow-interval', 'n_intervals'),
    State('cached-data', 'data')
)
def update_table(_, data):
    return [] if not data else get_statistical_data(pd.read_json(data)).to_dict('records')


@app.callback(
    Output('cached-patient-data', 'data'),
    [Input('quick-interval', 'n_intervals'),
    Input('patients-tabs', 'value')]
)
def cached_patient_data(_, patient_id):
    return get_patient_data(patient_id).to_json()


@app.callback(
    [Output('sensor-value', 'value'),
     Output('sensor-value', 'label')],
    [Input('cached-patient-data', 'modified_timestamp'),
     Input('sensors-tabs', 'value')],
    [State('cached-patient-data', 'data')]
)
def update_gauge(ts, sensor_name, data):
    if ts is None:
        raise PreventUpdate
    df = pd.read_json(data)
    value = df[df['name'] == sensor_name].iloc[0]['value']
    return value, f'Sensor Value: {value}'


@app.callback(
    Output('patient-name', 'children'),
    Input('patients-tabs', 'value')
)
def update_name(id):
    data = db.xrevrange(id, count=1)
    data_json = json.loads(data[0][1]['data'])
    return [html.H2(f'{data_json["firstname"]} {data_json["lastname"]} ({data_json["birthdate"]})')]


@app.callback(
    [Output('anomaly-graph', 'children')],
    [Input('cached-patient-data', 'modified_timestamp'),
     Input('sensors-tabs', 'value')],
    State('cached-patient-data', 'data')
)
def get_anomaly_graph(ts, sensor_name, data):
    if ts is None:
        raise PreventUpdate
    df = pd.read_json(data)

    df = df[(df['name'] == sensor_name) & (df['anomaly'] == True)]

    if (df.shape[0] == 0):
        return [html.H3('no anomaly recorded yet')]

    fig = px.scatter(df[['date', 'value']], x="date", y="value")
    return [dcc.Graph(figure=fig)]


@app.callback(
    Output('sensors-graph', 'figure'),
    [Input('cached-patient-data', 'modified_timestamp'),
     Input('sensors-graph-filter', 'value'), 
     Input('sensors-graph-time-filter', 'value')],
     State('cached-patient-data', 'data')
)
def get_sensors_graph(ts, sensors, time_range, data):
    if ts is None:
        raise PreventUpdate
    df = pd.read_json(data)
    df = df.loc[df['name'].isin(sensors)]

    now = datetime.utcnow()
    start_time = pd.to_datetime(now+timedelta(minutes=time_range[0]))
    end_time = pd.to_datetime(now+timedelta(minutes=time_range[1]))
    df = df.loc[(df['date'] > start_time) & (df['date'] < end_time)]
    fig = {'data': [{'x': df[df['name'] == sensor]['date'].tolist(), 'y': df[df['name'] == sensor]['value'].tolist(), 'name': sensor}
                    for sensor in sensors]}
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')
