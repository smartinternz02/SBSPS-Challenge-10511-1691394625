from flask import Flask, request, render_template
import pickle
from datetime import datetime, timedelta
import requests

app = Flask(__name__)
model = pickle.load(open('save_model.pkl', 'rb'))

def cal_pth(weather_data):
    total_precip = 0
    total_temp = 0
    total_humidity = 0
    for data in weather_data:
        if data['precip'] is not None or data['precip'] is not None or data['precip'] is not None:
            total_precip += data['precip']
            total_temp += data['temp']
            total_humidity += data['humidity']
    return (total_precip), total_temp / 90, total_humidity / 90

def expecting_crop_proba(crop_prob, expect):
    crop_label = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    idx = crop_label.index(expect.lower())
    total = 0
    for prob in crop_prob:
        total = max(prob[idx], total)
    return total

def calculate_data(n, p, k, ph, loc, expecting):
    try:
        now = datetime.now()
        forecast_start = datetime.strftime(now, "%Y-%m-%d")
        end = now + timedelta(days=75)
        forecast_end = datetime.strftime(end, "%Y-%m-%d")
        start1 = now - timedelta(days=365)
        year_start1 = datetime.strftime(start1, "%Y-%m-%d")
        end1 = start1 + timedelta(days=75)
        year_end1 = datetime.strftime(end1, "%Y-%m-%d")
        start2 = now - timedelta(days=(365*2))
        year_start2 = datetime.strftime(start2, "%Y-%m-%d")
        end2 = start2 + timedelta(days=75)
        year_end2 = datetime.strftime(end2, "%Y-%m-%d")
        link = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        field = "?unitGroup=metric&elements=datetime%2Ctemp%2Chumidity%2Cprecip%2Cpreciptype&include=days&"
        key = "key=X8G9TEUX485W65FLUXZ6AGDW9&contentType=json"
        API1 = link + loc + "/" + forecast_start + "/" + forecast_end + field + key
        API2 = link + loc + "/" + year_start1 + "/" + year_end1 + field + key
        API3 = link + loc + "/" + year_start2 + "/" + year_end2 + field + key
        response1 = requests.get(API1)
        response2 = requests.get(API2)
        response3 = requests.get(API3)
        weather_data1 = response1.json()['days']
        weather_data2 = response2.json()['days']
        weather_data3 = response3.json()['days']
        prec1, temp1, humd1 = cal_pth(weather_data1)
        prec2, temp2, humd2 = cal_pth(weather_data2)
        prec3, temp3, humd3 = cal_pth(weather_data3)
        crop_pred = []
        crop_pred.append(model.predict([[float(n), float(p), float(k), temp1, humd1, float(ph), prec1]])[0].capitalize())
        crop_pred.append(model.predict([[float(n), float(p), float(k), temp2, humd2, float(ph), prec2]])[0].capitalize())
        crop_pred.append(model.predict([[float(n), float(p), float(k), temp3, humd3, float(ph), prec3]])[0].capitalize())
        crop_prob = []
        crop_prob.append(model.predict_proba([[n, p, k, temp1, humd1, ph, prec1]])[0])
        crop_prob.append(model.predict_proba([[n, p, k, temp2, humd2, ph, prec2]])[0])
        crop_prob.append(model.predict_proba([[n, p, k, temp3, humd3, ph, prec3]])[0])
        crop_percentages = {}
        for crop in crop_pred:
            if crop in crop_percentages:
                crop_percentages[crop] += 1
            else:
                crop_percentages[crop] = 1
        expecting_proba = expecting_crop_proba(crop_prob, expecting)
        return crop_percentages, expecting_proba
    except:
        return False, False
        
@app.route('/',methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        nitrogen = float(request.form['Nitrogen'])
        phosphorous = float(request.form['Phosphorous'])
        potassium = float(request.form['Potassium'])
        ph = float(request.form['PH'])
        loc = request.form['loc']   
        expecting = request.form.get('expect')
        crop_rec, expecting_proba = calculate_data(nitrogen, phosphorous, potassium, ph, loc, expecting)
        if crop_rec is not False:
        #return str(nitrogen) + " " + str(phosphorous) + " " + str(potassium) + " " + str(ph) + " " + loc + " " + expecting
            return render_template("index.html", crop_rec=crop_rec, prob=(expecting_proba * 100), expecting=expecting)
        else:
            return render_template("index.html", info="API Calls exhuasted. Try again tomorrow.")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
