import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'Nitrogen':90 ,'Phosphorous':42 ,'Potassium':43 ,'Temparature':20.879744 ,'Humidity': 75,'PH':5.5, 'Rainfall':220 })

print(r.json())
