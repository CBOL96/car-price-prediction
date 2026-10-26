# Car Price Prediction

An end-to-end machine learning project that predicts the price of a car based on its technical specifications and characteristics.

The project covers the complete workflow: data cleaning, exploratory data analysis, feature preprocessing, model training and evaluation, model persistence, API development, and Docker containerization.

## Project Workflow

1. Download the dataset using `kagglehub`.
2. Clean and preprocess the data.
3. Perform exploratory data analysis.
4. Train and compare several regression models.
5. Select the best-performing model.
6. Save the trained pipeline.
7. Serve predictions through a Flask API.
8. Run the application in Docker.

## Models Compared

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- XGBoost Regressor

XGBoost achieved the best performance among the tested models, with an RMSE of approximately 8,050 and an R² score of 0.78 on the test set.

## Project Structure

```text
car-price-prediction/
├── train.py
├── prediction.py
├── prediction-test.py
├── requirements.txt
├── Dockerfile
├── notebooks/
│   ├── car-price-prediction.ipynb
│   └── prediction-test.ipynb
└── data/
    └── README.md
```

## How to Run Locally

1. Clone the repository:

```bash
git clone [https://github.com/your-username/car-price-prediction.git](https://github.com/your-username/car-price-prediction.git)
cd car-price-prediction
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python train.py
```

5. Start the prediction API:

```bash
python prediction.py
```

6. In another terminal, test the API:

```bash
python prediction-test.py
```

## How to Run with Docker

Build the image:

```bash
docker build -t car-price-prediction .
```

Run the container:

```bash
docker run -p 9696:9696 car-price-prediction
```

Test the API:

```bash
python prediction-test.py
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- Flask
- Docker
- Jupyter Notebook