#Libraries
import numpy as np
import pandas as pd
import kagglehub

# Download latest version of the dataset and load it into a pandas DataFrame
print("Downloading dataset...")
path = kagglehub.dataset_download("deepcontractor/car-price-prediction-challenge")
df = pd.read_csv(path + "/car_price_prediction.csv")

print("starting data preprocessing...")
#normalize the column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

#Normalize the string columns
string_columns = list(df.select_dtypes(include="object").columns)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")


#drop the rows with missing values in the levy column
df["levy"] = df["levy"].replace("-",np.nan)
df = df.dropna(subset=["levy"])
df["levy"] = df["levy"].astype(float)

#change to dummies the leather interior column
df["leather_interior"] = (
    df["leather_interior"]
    .replace({"yes": 1, "no": 0})
    .astype(int)
)

#Engine volume column, we will convert it to numeric
df["turbo"] = df["engine_volume"].str.contains("turbo", case=False, na=False).astype(int)
df["engine_volume"] = df["engine_volume"].str.replace("_turbo", "", case=False).astype(float)

#mileage column, we will convert it to numeric
df["mileage"] = df["mileage"].str.replace("_km", "", case=False).astype(float)

# Doors column
df["doors"] = df["doors"].replace({"04-may":"04-05","02-mar":"02-03"})


#Drop duplicates
df = df.drop_duplicates()

#we found that there are many cars with price less than 1000, which is not realistic. We will remove those rows from the dataset
df = df[df["price"] >= 1000].copy()

#there are some cars with engine volume equal to 0, which is not realistic. We will remove those rows from the dataset
df = df[df["engine_volume"] > 0].copy()


# Create numerical and Categorical Features
num_features = [
    "price",
    "levy",
    "prod._year",
    "engine_volume",
    "mileage",
    "cylinders",
    "airbags",
    "leather_interior",
    "turbo"
]

categorical_features = [
    "manufacturer",
    "model",
    "category",
    "fuel_type",
    "gear_box_type",
    "drive_wheels",
    "doors",
    "wheel",
    "color"
]
print("Data preprocessing completed.")

# Train Model
num_features.remove("price")

X = df.drop(columns=["id","price"])
y = df["price"]

#Preprocessing Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer,TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer, TargetEncoder, OneHotEncoder

numeric_pipeline = Pipeline([
        ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline([
    ('onehotencoder', OneHotEncoder(handle_unknown="ignore",drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, num_features),
    ("categorical", categorical_pipeline, categorical_features)
], remainder="drop")


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone

print("Training models...")
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(),
    'XGB Regressor': XGBRegressor(n_estimators=200)
}


target_transformer = PowerTransformer(method='yeo-johnson', standardize=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

for name, model in models.items():
    transformed_clone = clone(target_transformer)
    ttr = TransformedTargetRegressor(regressor=model, transformer=transformed_clone)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", ttr)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "R-squared": r2
    })

results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
print(results_df)


# save the best model
from joblib import dump
model = models['XGB Regressor']
ttr = TransformedTargetRegressor(regressor=model, transformer=target_transformer)
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ttr)
    ])
pipeline.fit(X_train, y_train)
dump(pipeline, "modelo_car_prediction.joblib")
print("Best model saved as modelo_car_prediction.joblib")



