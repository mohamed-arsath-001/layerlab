from fastapi.testclient import TestClient
import json
import os
import sys

# Import our FastAPI app
from api import app

client = TestClient(app)

def test_api():
    # Test Health
    print("Testing /health...")
    resp = client.get("/health")
    assert resp.status_code == 200
    print("Health OK:", resp.json())

    # Test Probe Pair
    print("\nTesting /api/probe-pair...")
    pwd = os.getcwd()
    path_a = os.path.join(pwd, "models", "modelA.safetensors").replace("\\", "/")
    path_b = os.path.join(pwd, "models", "modelB.safetensors").replace("\\", "/")
    
    payload = {"path_a": path_a, "path_b": path_b}
    resp = client.post("/api/probe-pair", json=payload)
    if resp.status_code != 200:
        print("Failed Probe Pair:", resp.text)
    assert resp.status_code == 200
    data = resp.json()
    print(f"Probe Pair OK. Found {data['common_count']} common keys.")
    
    # Test WebSocket Merge
    print("\nTesting WebSocket /ws/merge...")
    output_path = os.path.join(pwd, "output", "merged.safetensors").replace("\\", "/")
    
    try:
        with client.websocket_connect("/ws/merge") as ws:
            # Server sends connection ready
            msg = ws.receive_json()
            print("WS Handshake:", msg)

            merge_req = {
                "path_a": path_a,
                "path_b": path_b,
                "output_path": output_path,
                "algorithm": "lerp",
                "global_alpha": 0.5,
                "trim_fraction": 0.8,
                "warn_threshold": 0.7
            }
            ws.send_json(merge_req)
            
            while True:
                evt = ws.receive_json()
                print(f"WS Event [{evt.get('event_type')}]: {evt.get('message') or evt.get('layer_id') or ''}")
                if evt.get("event_type") == "error":
                    print("Error during merge!", evt)
                    break
                if evt.get("event_type") == "merge_complete":
                    print("Merge output saved successfully to:", evt.get("output_path"))
                    break
    except Exception as e:
        print(f"WebSocket test failed: {e}")

if __name__ == "__main__":
    test_api()
