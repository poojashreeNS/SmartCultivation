'''
    Smart Cultivation App

    @authors :
        Poojashree.NS : poojashree.ns@sjsu.edu
        Abraham Kong : Abraham.JKong@gmail.com
        Nisha Mohan Devadiga : nishamohan.devadiga@sjsu.edu
'''
import os
import dash
import pickle
import base64

import html as html
import numpy as np
import pandas as pd
from dash import dcc
from dash import html
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State


CROP_IMG_PATH = 'assets/images/'
DATA_PATH = 'Crop_Soil_Dataset.csv'
TRAINED_MODEL_PATH = 'Voting_Based_Model_Crop_Prediction_final.sav'
COLUMN_TRANSFORMER_PATH = 'colum_transformer.sav'
ENCODER_MODEL_PATH = 'label_encoder.sav'

crop_img_files = [os.path.join(CROP_IMG_PATH, f) for f in os.listdir(CROP_IMG_PATH)]


def model_inference(feature_arr):
    with open(TRAINED_MODEL_PATH, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
        prediction = model.predict(feature_arr)
        return prediction


def columntransformer_inference(user_input):
    transformer_model = pickle.load(open(COLUMN_TRANSFORMER_PATH, 'rb'))
    print(transformer_model)
    print(user_input)
    transformed_input = transformer_model.transform(user_input)
    print(transformed_input)
    return transformed_input


def get_crop_name(pred_int):
    label_model = pickle.load(open(ENCODER_MODEL_PATH, 'rb'))
    pred_crop_name = label_model.inverse_transform(pred_int)
    return pred_crop_name


def get_img_file(prediction, crops_list=crop_img_files):
    crop_name = prediction
    print(crop_name)
    img_file_raw = [f for f in crops_list if crop_name in f][0]
    img_file = img_file_raw.split('\\')
    return img_file[0]


def get_image(img_path):
    encoded_image = base64.b64encode(open(img_path, 'rb').read())
    img_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return img_src


def bi_fig(df):
    layout = go.Layout(
        margin={'l': 33, 'r': 40, 't': 20, 'b': 10},
    )

    fig = go.Figure()

    # Add Traces:
    fig.add_trace(go.Box(x=df['label'], y=df['N']))
    fig.add_trace(go.Box(x=df['label'], y=df['K'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['P'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['ph'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['temperature'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['humidity'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['rainfall'], visible=False))
    fig.add_trace(go.Box(x=df['label'], y=df['soil'], visible=False))

    # Add Buttons:
    fig.update_layout(
        updatemenus=[
            dict(
                active=1,
                buttons=list([

                    dict(label='Nitrogen - N',
                         method='update',
                         args=[{'visible': [True, False, False, False, False, False, False, False]},
                               {'title': 'Soil Properties - Nitrogen'}]),

                    dict(label='Phosphorous - K',
                         method='update',
                         args=[{'visible': [False, True, False, False, False, False, False, False]},
                               {'title': 'Soil Properties - Potassium'}]),

                    dict(label='Potassium - P',
                         method='update',
                         args=[{'visible': [False, False, True, False, False, False, False, False]},
                               {'title': 'Soil Properties - Phosphorus'}]),

                    dict(label='ph',
                         method='update',
                         args=[{'visible': [False, False, False, True, False, False, False, False]},
                               {'title': 'Soil Properties - pH'}]),

                    dict(label='Temperature',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, True, False, False]},
                               {'title': 'Climate Properties - Temperature'}]),

                    dict(label='Humidity',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, False, True, False]},
                               {'title': 'Climate Properties - Humidity'}]),

                    dict(label='Rainfall',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, False, False, True]},
                               {'title': 'Climate Properties - Rainfall'}]),

                    dict(label='Soil',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, False, False, True]},
                               {'title': 'Soil Properties - soil Types'}]),           

                ]),
            )
        ])
    # Set title:
    fig.update_layout(title_text='Bi-Variate Visualization', template='plotly_dark', height=500, width=900)
    return fig


def uni_fig(df):
    layout = go.Layout(
        margin={'l': 33, 'r': 40, 't': 20, 'b': 10},
    )
    fig = go.Figure()

    # Add trace
    fig.add_trace(go.Histogram(x=df['N']))
    fig.add_trace(go.Histogram(x=df['K']))
    fig.add_trace(go.Histogram(x=df['P']))
    fig.add_trace(go.Histogram(x=df['temperature']))
    fig.add_trace(go.Histogram(x=df['humidity']))
    fig.add_trace(go.Histogram(x=df['ph']))
    fig.add_trace(go.Histogram(x=df['rainfall']))
    fig.add_trace(go.Histogram(x=df['soil']))

    # Add Buttons
    fig.update_layout(
        updatemenus=[
            dict(
                active=1,
                buttons=list([

                    dict(label='N',
                         method='update',
                         args=[{'visible': [True, False, False, False, False, False, False, False]},
                               {'title': 'Nitrogen'}]),

                    dict(label='Potassium - K',
                         method='update',
                         args=[{'visible': [False, True, False, False, False, False, False, False]},
                               {'title': 'Potassium'}]),

                    dict(label='Phosphorus - P',
                         method='update',
                         args=[{'visible': [False, False, True, False, False, False, False, False]},
                               {'title': 'Phosphorus'}]),

                    dict(label='Temperature',
                         method='update',
                         args=[{'visible': [False, False, False, True, False, False, False, False]},
                               {'title': 'Temperature'}]),

                    dict(label='Humidity',
                         method='update',
                         args=[{'visible': [False, False, False, False, True, False, False, False]},
                               {'title': 'Humidity'}]),

                    dict(label='ph',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, True, False, False]},
                               {'title': 'ph'}]),

                    dict(label='Rainfall',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, False, True, False]},
                               {'title': 'Rainfall'}]),

                    dict(label='Soil',
                         method='update',
                         args=[{'visible': [False, False, False, False, False, False, False, True]},
                               {'title': 'Soil'}]),           

                ]),
            )
        ])

    # Set title
    fig.update_layout(title_text='Uni-variate Visualization', template='plotly_dark', width=900, height=500)

    return fig


# Initialise the app
app = dash.Dash(__name__)

server = app.server

# Define the app
app.layout = html.Div(children=[
    html.Div(className='row',
             children=[
                 html.Div(className='four columns div-user-controls',
                          children=[
                              html.H2(
                                  'Smart Cultivation application helps farmers to grow different crops depending on soil and climate parameters',
                                  style={'font-size': '20px'}),
                              html.Br(),
                              html.Div([
                                  html.H2('Nitrogen content in soil:', style={'font-size': '15px'}),

                                  dcc.Input(id="N",
                                            placeholder='Enter Nitrogen content in soil...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Phosphorous content in soil:', style={'font-size': '15px'}),
                                  dcc.Input(id="P",
                                            placeholder='Enter Phosphorous content in soil...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Potassium content in soil:', style={'font-size': '15px'}),
                                  dcc.Input(id="K",
                                            placeholder='Enter Potassium content in soil...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Temperature in °C:', style={'font-size': '15px'}),
                                  dcc.Input(id="temp",
                                            placeholder='Enter Temp in °C...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Humidity in %:', style={'font-size': '15px'}),
                                  dcc.Input(id="hum",
                                            placeholder='Enter Humidity in %...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('PH value of the soil (between 2-9):', style={'font-size': '15px'}),
                                  dcc.Input(id="ph",
                                            placeholder='Enter PH value of the soil...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Rainfall in mm:', style={'font-size': '15px'}),
                                  dcc.Input(id="rain",
                                            placeholder='Enter rainfall in mm...',
                                            type='text',
                                            persistence=False,
                                            style={'width': '400px'}
                                            )]),
                              html.Div([
                                  html.H2('Soil Type:', style={'font-size': '15px'}),
                                  dcc.Dropdown(id="soil",
                                               placeholder='Select soil type...',
                                               options=[
                                                   {'label': 'Black', 'value': 'black'},
                                                   {'label': 'Clayey', 'value': 'clayey'},
                                                   {'label': 'Loamy', 'value': 'loamy'},
                                                   {'label': 'Red', 'value': 'red'},
                                                   {'label': 'Sandy', 'value': 'sandy'}],
                                               style={'width': '400px'}
                                               )]),
                              html.Br(), html.Br(),
                              dcc.Store(id='store_inputs'),
                              html.Button('Submit', id='submit_button', n_clicks=0, disabled=False,
                                          style={'font-size': '15px',
                                                 'cursor': 'pointer',
                                                 'text-align': 'center',
                                                 'color': 'white',
                                                 }
                                          ),

                          ]),

                 html.Div(className='eight columns div-for-charts bg-black',
                          children=[
                              html.H2('Smart Cultivation', style={'text-align': 'center', "padding-top": "10px",
                                                                  'font-size': '35px', 'color': 'white'}),

                              html.H2('Data visualization:', style={"padding-top": "80px",
                                                                    "padding-left": "0", 'font-size': '25px'
                                                                    }),

                              html.Div([
                                  dcc.Dropdown(
                                      id="drop_down",
                                      options=[
                                          {'label': 'Uni-variate', 'value': 'feature'},
                                          {'label': 'Bi-variate', 'value': 'features'},
                                      ],
                                      style={'height': 30, 'width': 600},
                                      value='features',
                                      clearable=False)
                                      ]),
                              html.Br(),
                              html.Div([
                                  dcc.Graph(id='data_visualization',
                                            config={'displaylogo': False},
                                            style={'height': 550, 'width': 1200},
                                            )
                              ]),

                              html.Div([
                                  html.Div([
                                      html.P('Crop you can cultivate is:',
                                              style={"padding-top": "0px", 'font-size': '25px'}),

                                      html.Img(id="prediction_image")

                                  ], className="six columns"),

                                  html.Div(id='crop_name', className="six columns"),
                              ], className="row"),

                          ]),
                 html.Br(), html.Br(), html.Br()

             ])
])


@app.callback(Output("data_visualization", "figure"),
              Input('drop_down', 'value'),
              )
def dropdown_options(drop_value):
    df = pd.read_csv(DATA_PATH)
    df.soil = pd.Categorical(df.soil).codes

    fig_feature = uni_fig(df)
    fig_features = bi_fig(df)

    if drop_value == 'feature':
        return fig_feature

    if drop_value == 'features':
        return fig_features

    else:
        dash.no_update


@app.callback(Output("store_inputs", "data"),
              [Input('N', 'value'),
               Input("P", "value"),
               Input("K", "value"),
               Input("temp", "value"),
               Input("hum", "value"),
               Input("ph", "value"),
               Input("rain", "value"),
               Input("soil", "value")])
def store_inputs(N, P, K, temp, hum, ph, rain, soil):
    features_str = [N, P, K, temp, hum, ph, rain, soil]
    print(features_str)
    if len(features_str) == 8 and None not in features_str and '' not in features_str:
        return {'N': N, 'P': P, 'K': K, 'temp': temp, 'hum': hum, 'ph': ph, 'rain': rain, 'soil': soil}


@app.callback([Output("prediction_image", "src"),
               Output('crop_name', 'children')],
              Input('submit_button', 'n_clicks'),
              State('store_inputs', 'data'))
def update_crop_name(click, stored_inputs):
    cat_vars = ["soil"]
    num_vars = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if stored_inputs is not None:
        N = float(stored_inputs['N'])
        P = float(stored_inputs['P'])
        K = float(stored_inputs['K'])
        temp = float(stored_inputs['temp'])
        hum = float(stored_inputs['hum'])
        ph = float(stored_inputs['ph'])
        rain = float(stored_inputs['rain'])
        soil = stored_inputs['soil']

        tobeprocessed_input = np.array([[N, P, K, temp, hum, ph, rain, soil]])
        print(tobeprocessed_input)
        tobeprocessed_input = pd.DataFrame(tobeprocessed_input, columns=num_vars + cat_vars)
        print(tobeprocessed_input)

        processed_input = columntransformer_inference(tobeprocessed_input)
        print(processed_input)        

        prediction = model_inference(processed_input)
        predicted_crop_name = get_crop_name(prediction)[0]

        crop_img_file = get_img_file(predicted_crop_name)
        fig = get_image(crop_img_file)
        if 'submit_button' in trigger:
            return [fig,
                    html.P([f'{predicted_crop_name.capitalize()}',
                            ],
                           style={"padding-top": "10px",
                                  'display': 'list',
                                  'font-size': '25px', }),
                    ]
        else:
            return dash.no_update
    else:
        return dash.no_update


@app.callback([Output('N', 'value'),
               Output("P", "value"),
               Output("K", "value"),
               Output("temp", "value"),
               Output("hum", "value"),
               Output("ph", "value"),
               Output("rain", "value"),
               Output("soil", "value")],

              Input('submit_button', 'n_clicks'))
def reset_inputs(click):
    trigger = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit_button' in trigger:
        return [''] * 8
    else:
        return dash.no_update


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
