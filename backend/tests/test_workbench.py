"""Integration checks for real checkpoint loading and web workbench endpoints."""
import base64
import io
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c

def test_checkpoint_heads_and_real_prediction(client):
    metadata = client.get('/api/v1/model/architecture').json()
    assert metadata['checkpoint_status'] == 'All checkpoint keys matched'
    assert 0 < metadata['trainable_params'] < metadata['total_params']
    sample = client.get('/api/v1/samples/portrait-1')
    assert sample.status_code == 200
    result = client.post('/api/v1/predict', files={'file': ('sample.jpg', sample.content, 'image/jpeg')}).json()
    assert result['success']
    # This checkpoint's known portrait used to return zero with skipped output weights.
    assert 45 < result['faces'][0]['estimated_age'] < 52
    assert result['faces'][0]['gender_probability'] > 0.9

def test_explanations_cleanup_and_prediction_stability(client):
    sample = client.get('/api/v1/samples/portrait-1').content
    for target in ['age', 'gender']:
        response = client.post('/api/v1/explain', params={'target': target}, files={'file': ('sample.jpg', sample, 'image/jpeg')})
        assert response.status_code == 200, response.text
        for key in ['original', 'heatmap', 'overlay']:
            encoded = response.json()[key].split(',', 1)[1]
            image = Image.open(io.BytesIO(base64.b64decode(encoded)))
            assert image.size == (224, 224)
    result = client.post('/api/v1/predict', files={'file': ('sample.jpg', sample, 'image/jpeg')}).json()
    assert 45 < result['faces'][0]['estimated_age'] < 52

def test_archive_and_invalid_explanations(client):
    assert client.get('/api/v1/evaluation').json()['reports']
    assert client.get('/api/v1/samples/unknown').status_code == 404
    assert client.post('/api/v1/explain', files={'file': ('bad.txt', b'bad', 'text/plain')}).status_code == 400
    assert client.post('/api/v1/explain', files={'file': ('bad.jpg', b'bad', 'image/jpeg')}).status_code == 422
