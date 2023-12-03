#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils import leaf_disease
from utils import fertilizer_prediction
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9


# In[2]:


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
       'Blueberry___healthy','Cherry_(including_sour)___healthy','Cherry_(including_sour)___Powdery_mildew',
       'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
       'Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Grape___Black_rot','Grape___Esca_(Black_Measles)',
       'Grape___healthy','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot',
       'Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
       'Potato___healthy','Potato___Late_blight','Raspberry___healthy','Soybean___healthy',
       'Squash___Powdery_mildew','Strawberry___healthy','Strawberry___Leaf_scorch','Tomato___Bacterial_spot',
       'Tomato___Early_blight','Tomato___healthy','Tomato___Late_blight','Tomato___Leaf_Mold',
       'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
       'Tomato___Tomato_mosaic_virus','Tomato___Tomato_Yellow_Leaf_Curl_Virus']


# In[3]:


from utils import model
disease_model_path = 'C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\backend\\leaf_disease.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


    # Loading crop recommendation model

crop_recommendation_model_path = 'C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\pickle file\\Crop.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# In[ ]:


# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=leaf_disease):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    #img_u = img_t

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'AgriGrow - Home'
    return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\index.html', title=title)

# render crop recommendation form page


@ app.route('/Crop-Recommendation')
def crop_recommend():
    title = 'Agrigrow - Crop Recommendation'
    return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer--Recommendation')
def fertilizer_recommendation():
    title = 'AgriGrow - Fertilizer Suggestion'

    return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\fertilizer.html', title=title)

# render disease prediction input page




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'AgriGrow - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\try_again.html', title=title)

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'AgriGrow - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\Dataset\\Fertilizer_Prediction.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_prediction[key]))

    return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/leaf_disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'AgriGrow - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(leaf_disease[prediction]))
            return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('C:\\Users\\kanga\\OneDrive\\Desktop\\Minor Project\\frontend\\html\\disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:




