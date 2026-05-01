"""
Quick-start script: trains the model and launches the API server.
Usage: python run.py
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)


def main():
    artifacts_model = os.path.join(ROOT, 'artifacts', 'model.pt')

    if not os.path.exists(artifacts_model):
        print("=" * 50)
        print("No trained model found. Training now...")
        print("=" * 50)
        subprocess.run([sys.executable, '-m', 'model.train'], check=True)
        print()

    print("=" * 50)
    print("Starting FastAPI server on http://localhost:8000")
    print("Open the URL in your browser for the demo frontend.")
    print("API docs at http://localhost:8000/docs")
    print("=" * 50)
    subprocess.run([
        sys.executable, '-m', 'uvicorn',
        'webapi.main:app', '--host', '0.0.0.0', '--port', '8000', '--reload',
    ])


if __name__ == '__main__':
    main()
