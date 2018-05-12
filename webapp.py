from __future__ import unicode_literals

import pandas as pd
from flask import jsonify
from flask import Flask
from flask import render_template
from flask import request,flash

import mainFun

flask_app = Flask(__name__,static_url_path='',root_path='/Users/ValarMorghulis/Johnson_Johnson/interface')

mapptingPath = '/Users/ValarMorghulis/Johnson_Johnson/model/inputData/GMDN-UNSPSC-Mapping.csv'

## Main Page to Guide all users
@flask_app.route("/")
def index():
    # return render_template('landing_page.html', **context)
    return render_template('index.html')

## Data Understanding to Provide insights
@flask_app.route("/dataUnderstanding")
def dataUnderstanding():
    path = '/Users/ValarMorghulis/Johnson_Johnson/model/inputData/UNSPSC Full Data Update 2.csv'
    data = pd.read_csv(path)
    return render_template('dataUnderstanding.html')

## Modeling part
@flask_app.route("/modeling",methods=['GET', 'POST'])
def model():
    pred, GMDN_Des, Rec_UNSPSC, UNSPSC_Des = '', '', '', ''
    if request.method == 'POST':  # this block is only entered when the form is submitted
        columns = ['Breit', 'Brgew', 'Hoehe', 'Laeng', 'Volum', 'Zzwerks', 'Material Description', 'Gmdnptname',
                   'Minor_name']
        col = ['Material Description']

        if request.form.get('Breit'):
            dataSet = {}
            for i in columns:
                dataSet[i] = request.form.get(i)
            dataSet = pd.DataFrame([dataSet])
            dataSet = dataSet[columns]
            ## Test
            # dataSet = dataSet[col]
            model = mainFun.unspsc()
            pred = model.dataInput(dataSet)
            # pred = model.dataInput(dataSet)

        if request.form.get('GMDN Code'):
            gmdnCode = request.form.get("GMDN Code")
            gmdnCode = int(gmdnCode)
            print(gmdnCode)
            newData = pd.read_csv(mapptingPath, encoding='latin1')
            newData = newData[newData['GMDN'] == gmdnCode]
            GMDN_Des = newData.iloc[0, 2]
            Rec_UNSPSC = newData.iloc[0, 3]
            UNSPSC_Des = newData.iloc[0, 4]

        return render_template('model.html', say=str(pred), GMDN_Des=GMDN_Des,
                               Rec_UNSPSC=Rec_UNSPSC, UNSPSC_Des=UNSPSC_Des, scroll='something', scroll2='GMDN')

    return render_template('model.html')

## Interface Post and Get
@flask_app.route("/interface",methods=['GET', 'POST'])
def interfeace():
    pred, GMDN_Des, Rec_UNSPSC, UNSPSC_Des = '', '', '', ''
    if request.method == 'POST': #this block is only entered when the form is submitted
        columns = ['Breit', 'Brgew', 'Hoehe', 'Laeng', 'Volum', 'Zzwerks', 'Material Description', 'Gmdnptname',
                   'Minor_name']
        col = ['Material Description']

        if request.form.get('Breit'):
            dataSet = {}
            for i in columns :
                dataSet[i] = request.form.get(i)
            dataSet = pd.DataFrame([dataSet])
            dataSet = dataSet[columns]
            ## Test
            # dataSet = dataSet[col]
            model = mainFun.unspsc()
            pred = model.dataInput(dataSet)
            # pred = model.dataInput(dataSet)

        if request.form.get('GMDN Code'):
            gmdnCode = request.form.get("GMDN Code")
            gmdnCode =int(gmdnCode)
            print(gmdnCode)
            newData = pd.read_csv(mapptingPath, encoding='latin1')
            newData = newData[newData['GMDN'] == gmdnCode]
            GMDN_Des = newData.iloc[0,2]
            Rec_UNSPSC = newData.iloc[0,3]
            UNSPSC_Des = newData.iloc[0,4]

        return render_template('interface.html', say=str(pred), GMDN_Des =GMDN_Des,
                               Rec_UNSPSC=Rec_UNSPSC,UNSPSC_Des=UNSPSC_Des, scroll='something',scroll2='GMDN' )


        # print(dataSet)
        # return jsonify(result=dataSet)

    return render_template('interface.html')

if __name__ == "__main__":
    flask_app.run(debug=True)
