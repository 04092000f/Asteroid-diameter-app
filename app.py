from flask import Flask, make_response, request, jsonify
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle



app = Flask(__name__)

def transform(text_file_contents):
    return text_file_contents.replace("=", ",")


@app.route('/')
def form():
    return """
        <html>
            <body>
                <h1>Asteroid Diameter Prediction</h1>
                <br>
                <br>
                <p> Insert your CSV file and then download the Result
                <p> Make Sure that your dataset has atleast the following attributes
                <ul>
                <li>H: Absolute Magnitude Parameter{float value}</li>
                <li>n_obs_used: No of Radar Observations used{integer value}</li>
                <li>data_arc: Observation Arc in days{float value}</li>
                <li>albedo: Geometric Albedo{float value}</li>
                <li>a: Semi major Axis{float value}</li>
                <li>q: Perihelion Distance{float value}</li>
                <li>moid: Minimum Earth Orbit Intersection Distance{float value}</li>
                <li>neo: Near Earth Object{Categorical feature with either Yes or No}</li>
                <li>pha: Potentially Hazardous Asteroid{Categorical feature with either Yes or No}</li>
                </ul>
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>
                    </br>
                    </br>
                    <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>
                </form>
            </body>
        </html>
    """
@app.route('/transform', methods=["POST"])
def transform_view():
    f = request.files['data_file']
    if not f:
        return "No file"

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    csv_input = csv.reader(stream)
    #print("file contents: ", file_contents)
    #print(type(file_contents))
    print(csv_input)

    stream.seek(0)
    result = transform(stream.read())

    df = pd.read_csv(StringIO(result))
    # load the model from disk
    model = pickle.load(open('XGB_Updated_with_137681_pts.pkl', 'rb'))
    sd= pickle.load(open('Scalar.sav', 'rb'))
    rfmis=pickle.load(open('RF_for_data_arc.pkl','rb'))
    xH=pickle.load(open('XGB_for_H.pkl','rb'))
    lmoid=pickle.load(open('Linear_for_moid.pkl','rb'))
    ohe=pickle.load(open('One-Hot-Encoder.sav','rb'))
    x=df.copy()

    if x['ad'] is np.nan:
        # Keeping Default value of a as median 
        x['ad'].replace([np.nan], 3.03775, inplace=True)

    if x['albedo'] is np.nan:
        # Keeping Default value of a as median 
        x['albedo'].replace([np.nan], 0.078, inplace=True)

    if x['a'] is np.nan:
        # Keeping Default value of a as median 
        x['a'].replace([np.nan], 2.64421, inplace=True)
        

    if x['condition_code'] is np.nan:
        # Keeping Default value of condition_code as 0
        x['condition_code'].replace([np.nan], 0, inplace=True)

    if x['H'] is np.nan:
        yp = xH.predict(x[['q',"data_arc","n_obs_used"]])
        x.loc[x['H'].isna(),'H']=yp

    if x['neo'].isnull().values.any():
        if x['q']<=1.3:
            x['neo'] = x['neo'].replace(np.nan, 'Y')
        else:
            x['neo'] = x['neo'].replace(np.nan, 'N')

    if x['pha'].isnull().values.any():
        if (x['moid'].values<=0.05 and x['H'].values<=22):
            x['pha'] = x['neo'].replace(np.nan, 'Y')
        else:
            x['pha'] = x['pha'].replace(np.nan, 'N')

      

    if df['a'].values<-33000 or df['a'].values>4000:
        return jsonify({'Error':'Enter the value of Semi-Major axis between -33000 and 4000'})
        
    if df["q"].values<=0 or df["q"].values>81:
        return jsonify({'Error':'Enter the value of Perihelion Distance between 0 and 81'})

    if df["n_obs_used"].values<=0:
        return jsonify({'Error':'Enter the value of Radar Observations greater than 0'})

    if df["H"].values<-10 or df["H"].values>35:
        return jsonify({'Error':'Enter H between -10 and 35'})

    if df["albedo"].values<0 or df["albedo"].values>1:
        return jsonify({'Error':'Enter albedo between 0 and 1'})

    if df["moid"].values<=0 or df["moid"].values>80:
        return jsonify({'Error':'Enter moid between 0 and 80'})

    if df["data_arc"].values<0:
        return jsonify({'Error':'Enter data_arc value greater than 0'})

    x_norm=sd.transform(x[['a', 'q', 'data_arc', 'n_obs_used', 'H',
       'albedo', 'moid']])
    x_norm=pd.DataFrame(data=x_norm,columns=['a', 'q', 'data_arc', 'n_obs_used', 'H',
       'albedo', 'moid'])
    x_neo_encode=ohe.transform(x['neo'].values.reshape(-1,1))
    x_pha_encode=ohe.transform(x['pha'].values.reshape(-1,1))

    x=x_norm[['a','q','data_arc','n_obs_used','H','albedo','moid']]
    x['neo']=x_neo_encode
    x['pha']=x_pha_encode
    ypred = model.predict(x)
    return jsonify({'Diameter of the Asteroid is : ':str(ypred)})
    #response = make_response(a.to_csv(index=False))
    #response.headers["Content-Disposition"] = "attachment; filename=Predictions with R Squared and NMAE.csv"
    #return response

if __name__ == "__main__":
    app.run(debug=False,port=5000)
