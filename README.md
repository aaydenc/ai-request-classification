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

#### 5. Run the API

```bash
uvicorn scripts.api:app --reload
```

## Usage

### Import Data

To import data for the model to use run:

```bash
python scripts/import_data.py
```

### Train Model

To train and view predictions run after data has been imported run:

```bash
python scripts/model.py
```

### Testing API

Automated API tests are implementd using pytest and FastAPI TestClient.
To run the tests run:

```bash
python -m pytest -v
```

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

```json
{
  "predicted_category": "road",
  "confidence_score": 0.85
}
```
