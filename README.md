# House Price Prediction API

A machine learning deployment project that trains a **Linear Regression** model on the **California Housing dataset**, evaluates it using **Root Mean Squared Error (RMSE)**, saves the trained model, and exposes it through a REST API using **FastAPI** and **Docker**.

The project demonstrates the process of taking a machine learning model from training to a containerized prediction service.

## 1. Project Overview

```text
California Housing Dataset
          |
          v
     Select Features
          |
          v
     Train / Test Split
         80% / 20%
          |
          v
   Linear Regression
          |
          v
    RMSE Evaluation
          |
          v
     Save Model
       .joblib
          |
          v
       FastAPI
      /predict
          |
          v
        Docker
          |
          v
   Prediction Service
```

## 2. Dataset

The project uses the **California Housing dataset** provided by scikit-learn.

Five features are selected:

| Feature     | Description                  |
| ----------- | ---------------------------- |
| `MedInc`    | Median income in block group |
| `Latitude`  | Geographic latitude          |
| `Longitude` | Geographic longitude         |
| `AveRooms`  | Average rooms per household  |
| `HouseAge`  | Median house age             |

The target is `MedHouseVal`, which represents the median house value in **hundreds of thousands of dollars**.

Examples:

* `1.0` → $100,000
* `4.0` → $400,000
* `10.0` → $1,000,000

## 3. Machine Learning Pipeline

### 3.1 Load the Dataset

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)

df = data["data"]
target = data["target"]
```

`as_frame=True` returns the features as a pandas DataFrame, allowing the features to be selected by column name.

### 3.2 Select Features

```python
selected_features = [
    "MedInc",
    "Latitude",
    "Longitude",
    "AveRooms",
    "HouseAge"
]

X = df[selected_features]
y = target
```

* `X` contains the five input features.
* `y` contains the target house values.

### 3.3 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is split into:

* **80% training data**
* **20% test data**

`random_state=42` makes the split reproducible.

### 3.4 Train the Linear Regression Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns the relationship between the five input features and the target house value.

### 3.5 Evaluate the Model

Generate predictions on the test set:

```python
predictions = model.predict(X_test)
```

Calculate RMSE:

```python
rmse = mean_squared_error(y_test, predictions) ** 0.5
print("RMSE:", rmse)
```

RMSE is calculated as:

```text
RMSE = sqrt(mean((y - ŷ)²))
```

RMSE measures the typical prediction error in the same units as the target, making it easier to interpret than MSE.

## 4. Saving the Model

The trained model is saved using Joblib:

```python
import joblib

joblib.dump(
    model,
    "model/linear_regression_model.joblib"
)
```

The `.joblib` file contains the trained Linear Regression model, including its learned coefficients and intercept.

The API loads this saved model instead of retraining it.

## 5. FastAPI

The FastAPI application loads the saved model:

```python
model = joblib.load(
    "model/linear_regression_model.joblib"
)
```

### 5.1 Health Endpoint

The application provides a health endpoint:

```text
GET /health
```

Test it with:

```bash
curl http://127.0.0.1:80/health
```

Response:

```json
{"status":"ok"}
```

### 5.2 Prediction Endpoint

The prediction endpoint is:

```text
POST /predict
```

Example request:

```bash
curl -X POST \
  http://127.0.0.1:80/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 4.5,
    "Latitude": 37.77,
    "Longitude": -122.42,
    "AveRooms": 6.0,
    "HouseAge": 30.0
  }'
```

FastAPI uses Pydantic to validate the incoming request.

The input is converted into a NumPy array:

```python
input_data = np.array([[
    data.MedInc,
    data.Latitude,
    data.Longitude,
    data.AveRooms,
    data.HouseAge
]])
```

The loaded model generates the prediction:

```python
prediction = model.predict(input_data)
```

The API returns the prediction as JSON:

```json
{
  "predicted_house_price": 4.2
}
```

A prediction of `4.2` represents approximately **$420,000**, because the California Housing target is measured in hundreds of thousands of dollars.

## 6. Docker

Docker packages the application, dependencies, saved model, and FastAPI server into a container.

The Docker image contains:

* Python
* FastAPI
* Uvicorn
* scikit-learn
* pandas
* NumPy
* Joblib
* Pydantic
* FastAPI application
* Trained model

### 6.1 Build the Docker Image

```bash
docker build -t house-price-prediction-api:v1 .
```

### 6.2 Run the Container

```bash
docker run -d -p 80:80 house-price-prediction-api:v1
```

The port mapping is:

```text
Host port 80 → Container port 80
```

The API is then available at:

```text
http://127.0.0.1:80
```

## 7. API Testing

The `test.sh` script can be used to test the running API.

Example:

```bash
curl -X POST \
  http://127.0.0.1:80/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 35,
    "Latitude": 34.05,
    "Longitude": -118.24,
    "AveRooms": 8.0,
    "HouseAge": 10.0
  }'
```

Example response:

```json
{
  "predicted_house_price": 13.596671738389922
}
```

This corresponds to approximately **$1.36 million**.

## 8. Technologies

* **Python** — application and machine learning code
* **scikit-learn** — dataset, train/test split, Linear Regression, and RMSE
* **pandas** — DataFrame returned by `fetch_california_housing(as_frame=True)`
* **NumPy** — preparing API input
* **Joblib** — model serialization
* **FastAPI** — REST API
* **Pydantic** — request validation
* **Uvicorn** — ASGI server
* **Docker** — containerization

## 9. End-to-End Workflow

1. Load dataset → 2. Select features → 3. Split 80/20 → 4. Train Linear Regression → 5. Evaluate with RMSE → 6. Save `.joblib` → 7. Build Docker image → 8. Start FastAPI container → 9. Load model → 10. POST `/predict` → 11. Generate and return prediction
