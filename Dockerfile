FROM python:3.12-slim-bookworm

#Working directory
WORKDIR /app

# copy and install dependencies
COPY requirements.txt ./

#Instalar dependencias
RUN python -mpip install --no-cache-dir -r requirements.txt

COPY ["prediction.py","modelo_car_prediction.joblib","./"]

#expose the app to port 9696
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "prediction:app"]