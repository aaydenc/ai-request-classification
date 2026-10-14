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

### Setup

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aaydenc/ai-request-classification.git
cd ai-request-classification
pip install -r requirements.txt
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