from fastapi.testclient import TestClient
from scripts.api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_categories():
    response = client.get("/categories")

    assert response.status_code == 200

    categories = response.json()["categories"]
    assert len(categories) > 0
    assert "road" in categories
    assert "safety" in categories
    assert "waste" in categories
    assert "water" in categories

def test_predict_road_request():
    response = client.post(
        "/predict",
        json={"request_text": "There is a pothole in the road"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "road"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_road_request_2():
    response = client.post(
        "/predict",
        json={"request_text": "There is a tree fallen in the road"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "road"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_safety_request():
    response = client.post(
        "/predict",
        json={"request_text": "Handrail on pedestrian bridge is missing"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "safety"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_safety_request_2():
    response = client.post(
        "/predict",
        json={"request_text": "Cracked glass at the bus stop is hazardous to the public"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "safety"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_waste_request():
    response = client.post(
        "/predict",
        json={"request_text": "Missed rubbish collection on my street"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "waste"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_waste_request_2():
    response = client.post(
        "/predict",
        json={"request_text": "There is a whole lot of rubbish left at the park"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "waste"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_water_request():
    response = client.post(
        "/predict",
        json={"request_text": "Sewer smell is really bad near the river"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "water"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_water_request_2():
    response = client.post(
        "/predict",
        json={"request_text": "Standing water near playground may be a slipping hazard"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_category"] == "water"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_empty_request_string():
    response = client.post(
        "/predict",
        json={"request_text": ""}
    )

    assert response.status_code == 422

def test_predict_empty_request():
    response = client.post(
        "/predict",
    )

    assert response.status_code == 422