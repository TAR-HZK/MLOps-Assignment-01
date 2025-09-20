import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.app import app  # make sure app/app.py defines `app = Flask(__name__)`

def test_index():
    client = app.test_client()
    resp = client.get('/')
    assert resp.status_code == 200
