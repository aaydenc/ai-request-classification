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
    assert isinstance(categories, list)

def test_predict_road_pothole_request():
    response = client.post(
        "/predict",
        json={"request_text": "There is a pothole in the road"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "There is a pothole in the road"
    assert result["predicted_category"] == "road"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_road_fallen_tree_request():
    response = client.post(
        "/predict",
        json={"request_text": "There is a tree fallen in the road"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "There is a tree fallen in the road"
    assert result["predicted_category"] == "road"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_safety_bridge_handrail_request():
    response = client.post(
        "/predict",
        json={"request_text": "Handrail on pedestrian bridge is missing"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "Handrail on pedestrian bridge is missing"
    assert result["predicted_category"] == "safety"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_safety_cracked_glass_request():
    response = client.post(
        "/predict",
        json={"request_text": "Cracked glass at the bus stop is hazardous to the public"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "Cracked glass at the bus stop is hazardous to the public"
    assert result["predicted_category"] == "safety"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_waste_rubbish_street_request():
    response = client.post(
        "/predict",
        json={"request_text": "Missed rubbish collection on my street"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "Missed rubbish collection on my street"
    assert result["predicted_category"] == "waste"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_waste_rubbish_park_request():
    response = client.post(
        "/predict",
        json={"request_text": "There is a whole lot of rubbish left at the park"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "There is a whole lot of rubbish left at the park"
    assert result["predicted_category"] == "waste"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_water_sewer_smell_request():
    response = client.post(
        "/predict",
        json={"request_text": "Sewer smell is really bad near the river"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "Sewer smell is really bad near the river"
    assert result["predicted_category"] == "water"
    assert 0 <= result["confidence_score"] <= 1

def test_predict_water_standing_water_request():
    response = client.post(
        "/predict",
        json={"request_text": "Standing water near playground may be a slipping hazard"}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["request_text"] == "Standing water near playground may be a slipping hazard"
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

def test_predict_whitespace_request():
    response = client.post(
        "/predict",
        json={"request_text": "   "}
    )

    assert response.status_code == 422

def test_predict_invalid_input_type():
    response = client.post(
        "/predict",
        json={"request_text": 123}
    )

    assert response.status_code == 422