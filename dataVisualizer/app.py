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
from datetime import datetime, timedelta
from dash.exceptions import PreventUpdate


app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500&display=swap"])
log = app.logger

db = redis.Redis(host='ppdv-redis',
                 port=6379, decode_responses=True)


def rms(x):
    return np.sqrt(sum(x**2/len(x)))


def quantile(x, q):
    return np.quantile(x, q)


def get_patient_data(id, last_ts):
    df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    if last_ts is not None:
        data = db.xrevrange(id, min=last_ts)
        if len(data) == 0:
            raise PreventUpdate
    else:
        data = db.xrevrange(id, count=10)
    for j in range(len(data)):
        data_json = json.loads(data[j][1]['data'])
        tmp = pd.json_normalize(data_json['trace']['sensors'])
        tmp['date'] = datetime.utcfromtimestamp(
            int(data[j][0].split('-')[0])//1000)
        df = df.append(tmp)
    df = df.reset_index(drop=True)
    return df, int(data[0][0].split('-')[0])+1


def get_statistical_data(df):
    df = df.groupby(['name']).agg({
        'value': [
            'min', 'max', 'mean', ("rms", rms),
            ("first quartile", lambda x: quantile(x, 0.25)),
            ("median", lambda x: quantile(x, 0.5)),
            ("third quartile", lambda x: quantile(x, 0.75))
        ]
    })
    df.columns = [t[1] for t in df.columns.values]
    df.insert(loc=0, column='name', value=['L0', 'L1', 'L2', 'R0', 'R1', 'R2'])
    return df


def get_patient_table_data(df):
    cols = ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
    new_df = pd.DataFrame(columns=cols)

    for col in cols:
        sensor_data = df[df['name'] == col][['value', 'anomaly']]
        new_df[col] = sensor_data['value'][:10].tolist()
        new_df[f'anomaly_{col}'] = sensor_data['anomaly'][:10].tolist()
    dates = df[df['name'] == 'L0'][['date']]['date'][:10].tolist()
    anomalies = df[df['name'] == 'L0'][['anomaly']]['anomaly'][:10].tolist()
    new_df['date'] = dates
    new_df['anomaly'] = anomalies
    return new_df


app.layout = dbc.Container([
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
    dcc.Store(id='last-cache'),
    dcc.Store(id='prev-patient'),
    dbc.Card([
        dbc.CardHeader([html.H1("All patients aggregated statistical data")]),
        dbc.CardBody([
            dash_table.DataTable(
                id='full-table',
                columns=[{'name': i, "id": i} for i in ['name', 'min',
                                                        'max', 'mean', 'rms', 'first quartile', 'median', 'third quartile']]
            )
        ])
    ]),

    dbc.Card(className='in-card', children=[
        dbc.CardHeader([
            dbc.Tabs(id="patients-tabs", active_tab='1', children=[
                dbc.Tab(label='Patient One', tab_id='1'),
                dbc.Tab(label='Patient Two', tab_id='2'),
                dbc.Tab(label='Patient Three', tab_id='3'),
                dbc.Tab(label='Patient Four', tab_id='4'),
                dbc.Tab(label='Patient Five', tab_id='5'),
                dbc.Tab(label='Patient Six', tab_id='6'),
            ]),
        ]),
        dbc.CardBody([
            html.Div(id='patient-name'),
            dash_table.DataTable(
                id='patient-table',
                columns=[{'name': i, "id": i}
                         for i in ["date", "L0", "L1", "L2", "R0", "R1", "R2", "anomaly_L0", "anomaly_L1", "anomaly_L2", "anomaly_R0", "anomaly_R1", "anomaly_R2"]],
                style_data_conditional=[*[
                    {
                        'if': {
                            'filter_query': '{anomaly_' + col + '} contains true',
                            'column_id': col
                        },
                        'backgroundColor': '#F5DDDC'
                    } for col in ["L0", "L1", "L2", "R0", "R1", "R2"]],
                    *[{
                        'if': {
                            'filter_query': '{anomaly_' + col + '} contains false',
                            'column_id': col
                        },
                        'backgroundColor': 'rgba(52, 179, 86, 0.3)'
                    } for col in ["L0", "L1", "L2", "R0", "R1", "R2"]]
                ],
                style_cell_conditional=[
                    {
                        'if': {'column_id': col},
                        'display': 'none'
                    } for col in ["anomaly_L0", "anomaly_L1", "anomaly_L2", "anomaly_R0", "anomaly_R1", "anomaly_R2"]
                ]
            ),
            html.Div(className='in-card', children=[
                html.H3("Sensors graph"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.Dropdown(id='sensors-graph-filter', options=[
                            {'value': x, 'label': x} for x in ['L0', 'L1', 'L2', 'R0', 'R1', 'R2']
                        ], multi=True, value=['L0', 'L1', 'L2', 'R0', 'R1', 'R2']),
                        dcc.RangeSlider(
                            id='sensors-graph-time-filter',
                            className='sensors-graph-time-filter',
                            min=-10,
                            max=0,
                            step=0.5,
                            value=[-10, 0],
                            marks={-10: "-10 min", -8: "-8 min", -6: "-6 min", -4: "-4 min", -2: "-2 min", 0: "now"}),
                        dcc.Graph(id='sensors-graph')
                    ])
                ])
            ]),
            dbc.Card(className="in-card", children=[
                dbc.CardHeader([
                    dbc.Tabs([
                        dbc.Tab(label='L0', tab_id='L0'),
                        dbc.Tab(label='L1', tab_id='L1'),
                        dbc.Tab(label='L2', tab_id='L2'),
                        dbc.Tab(label='R0', tab_id='R0'),
                        dbc.Tab(label='R1', tab_id='R1'),
                        dbc.Tab(label='R2', tab_id='R2'),
                    ], id="sensors-tabs", active_tab="L0"),
                ]),
                dbc.CardBody([
                    dbc.Row(
                        [
                            dbc.Col(daq.Gauge(
                                id='sensor-value',
                                label="Sensor Value:",
                                value=0,
                                min=0,
                                max=1023
                            ), md=4),
                            dbc.Col(html.Div(id='anomaly-graph'), md=8),
                        ],
                        align="center",
                    )
                ])
            ])
        ]),
    ])
])


@app.callback(
    Output('cached-data', 'data'),
    Input('quick-interval', 'n_intervals'),
    [State('last-cache', 'data'),
     State('cached-data', 'data')]
)
def cached_data(_, ts, old_df):
    if old_df is None:
        df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    else:
        df = pd.read_json(old_df)
    for i in range(1, 7):
        df = get_patient_data(i, ts)[0].append(df)
    df = df.reset_index(drop=True)
    if df.shape[0] > 10800:
        df = df[:10800]
    return df.to_json()


@app.callback(
    Output('patient-table', 'data'),
    [Input('patients-tabs', 'active_tab'),
     Input('cached-patient-data', 'modified_timestamp')],
    State('cached-patient-data', 'data')
)
def change_tab(_, ts, data):
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
    [Output('cached-patient-data', 'data'),
     Output('last-cache', 'data'),
     Output('prev-patient', 'data')],
    [Input('quick-interval', 'n_intervals'),
     Input('patients-tabs', 'active_tab')],
    [State('last-cache', 'data'),
     State('cached-patient-data', 'data'),
     State('prev-patient', 'data')]
)
def cached_patient_data(_, patient_id, last_ts, old_df, prev_patient):
    if prev_patient != patient_id:
        old_df = None
        prev_patient = patient_id
        last_ts = int(datetime.now().timestamp()*1000-10*60*1000)

    if old_df is None:
        df = pd.DataFrame(columns=["anomaly", "id", "name", "value"])
    else:
        df = pd.read_json(old_df)

    data, ts = get_patient_data(patient_id, last_ts)
    data = data.append(df)
    data = data.reset_index(drop=True)
    if data.shape[0] > 1800:
        data = data[:1800]

    return data.to_json(), ts, prev_patient


@app.callback(
    [Output('sensor-value', 'value'),
     Output('sensor-value', 'label')],
    [Input('cached-patient-data', 'modified_timestamp'),
     Input('sensors-tabs', 'active_tab')],
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
    Input('patients-tabs', 'active_tab')
)
def update_name(id):
    data = db.xrevrange(id, count=1)
    data_json = json.loads(data[0][1]['data'])
    return [html.H2(f'{data_json["firstname"]} {data_json["lastname"]} ({data_json["birthdate"]})')]


@app.callback(
    [Output('anomaly-graph', 'children')],
    [Input('cached-patient-data', 'modified_timestamp'),
     Input('sensors-tabs', 'active_tab')],
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
