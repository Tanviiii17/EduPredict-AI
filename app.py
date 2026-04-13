"""
Flask Application for Student Adaptability Level Prediction
Entry point for the AI Product Dashboard
"""
import sys
import os

# Add the project root to the python path to allow imports from scripts.*
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == '__main__':
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
