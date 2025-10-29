from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_ask():
    response = client.post("/ask", json={"question": "What is the capital of Morrowind?", "thread_id": "1"})
    assert response.status_code == 200
    assert "answer" in response.json()
