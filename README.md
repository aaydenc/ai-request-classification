# AI Public Request Classification

A simple logistic regression model designed to classify different public service requests.

## Features

- Importing and preprocessing data using pandas.
- Training a Logistic Regression classification model using TF-IDF.
- Saving and loading the trained model using joblib.
- Classifying public service requests into one of four categories.
- Providing predictions and confidence scores through a FastAPI backend.
- Providing API health and supported category endpoints.
- Automated API testing using pytest and FastAPI TestClient.

## Installation

### Requirements

- Python 3.12+
- pip
- git

### Setup

Clone the repository and install the required dependencies:

#### 1. Clone the repository

```bash
git clone https://github.com/aaydenc/ai-request-classification.git
cd ai-request-classification
```

#### 2. Create a virtual environment

```bash
python -m venv .venv
```

#### 3. Activate the virtual envrioment

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

#### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Import Data

Import and preprocess the raw dataset:

```bash
python scripts/import_data.py
```

This processes the raw dataset and saves the cleaned data for model training.

### Train Model

After importing the data, train the classification model:

```bash
python scripts/model.py
```

This trains the Logistic Regression pipeline, evaluates its performance and saves the trained model using joblib.

## Experiment with Model Configurations

To evaluate different model configurations and compare their performance, run:

```bash
python scripts/model_experiment.py
```

### Run the API

After the model has been trained and saved, start the FastAPI application:

```bash
uvicorn scripts.api:app --reload
```

The API will be available at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs

## API Endpoints

The API provides three endpoints:

| Method | Endpoint      | Description                                                                       |
| ------ | ------------- | --------------------------------------------------------------------------------- |
| GET    | `/health`     | Checks whether the API is running.                                                |
| GET    | `/categories` | Returns the supported request categories.                                         |
| POST   | `/predict`    | Predicts the category of a public service request and returns a confidence score. |

### Example Prediction Request

Send a POST request to `/predict` with the following JSON body:

```json
{
  "request_text": "There is a pothole developing in the road"
}
```

The API returns the predicted category and confidence score.

Example response:

```json
{
  "request_text": "There is a pothole developing in the road"
  "predicted_category": "road",
  "confidence_score": 0.85
}
```

Note: The confidence score shown is an illustrative example and may differ depending on the trained model.

## API Validation

The API validates incoming prediction requests. Invalid requests, such as empty request text, missing fields or whitespace-only input, return a 422 Unprocessable Entity response.

## Testing

Automated API tests are implemented using pytest and FastAPI TestClient.

The tests cover:

- API health checks.
- Supported request categories.
- Valid predictions across all four categories.
- Invalid prediciton requests and input validation.

To run the tests, execute:

```bash
python -m pytest -v
```
