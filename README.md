# AI Public Request Classification

A simple logistic regression model designed to classify different public service requests.

## Features

- Importing data into pandas dataframe.
- Training model with provided data.
- Outputs predicted request classifications to either the console or a file (defaults to /outputs/classification_report.txt)

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

#### 2. Create a virtual enviroment

```bash
python -m venv .venv
```

#### 3. Activate the virtual envrioment

Widnows PowerShell:

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
