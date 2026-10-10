import requests
url = 'http://localhost:9696/prediction'

customer = {
    "levy": 800.0,
    "manufacturer": "toyota",
    "model": "camry",
    "prod._year": 2017,
    "category": "sedan",
    "fuel_type": "petrol",
    "engine_volume": 2.5,
    "mileage": 85000.0,
    "cylinders": 4.0,
    "gear_box_type": "automatic",
    "drive_wheels": "front",
    "doors": "04-05",
    "wheel": "left_wheel",
    "color": "black",
    "airbags": 12,
    "leather_interior": 1,
    "turbo": 0
}
response = requests.post(url, json=customer).json()
print(response)