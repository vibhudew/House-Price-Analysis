from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# df=pd.read_csv("cluster_house123.csv")
# model=df.drop(['ID'],axis=1)

app=Flask(__name__)
model=pickle.load(open('xgbmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('price_predict.html')

'''
@app.route('/predict_price',methods=['POST'])
def predict_price():

    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    floors = request.form['floors']
    waterfront = request.form['waterfront']
    view = request.form['view']
    condition = request.form['condition']
    grade = request.form['grade']
    yr_built = request.form['yr_built']
    yr_renovated = request.form['yr_renovated']
    lat = request.form['lat']
    long = request.form['long']
    sqft_total = request.form['sqft_total']

    arr = np.array([['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view',
       'condition', 'grade', 'yr_built', 'yr_renovated', 'lat', 'long',
       'sqft_total']])

    pred=model.predict(arr)

    return render_template('price_predict.html',data=pred,bedrooms=bedrooms,bathrooms=bathrooms,floors=floors,waterfront=waterfront,
    view=view,condition=condition,grade=grade, yr_built =yr_built,yr_renovated=yr_renovated,lat=lat,long=long,sqft_total=sqft_total)
'''

#To use the predict button in our web-app
@app.route('/predictPrice',methods=['POST'])
def predictPrice():
    #For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0]) 
    return render_template('price_predict.html', prediction_text='Price in the house is :{}'.format(output),bedrooms=int_features[0],bathrooms=int_features[1],floors=int_features[2],waterfront=int_features[3],
                                     view=int_features[4],condition=int_features[5],grade=int_features[6], yr_built =int_features[7],yr_renovated=int_features[8],lat=int_features[9],long=int_features[10],sqft_total=int_features[11])

if __name__=="__main__":
    app.run(debug=True)
