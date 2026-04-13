from flask import Flask
from flask_cors import CORS
import os
import logging

def create_app():
    # Set up template and static folder to point to the app folder
    # Flask assumes templates and static are relative to the root path of the module
    app = Flask(__name__, template_folder='templates', static_folder='static')
    CORS(app)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Ensure directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Register blueprints
    from app.routes.api import api_bp
    from app.routes.views import views_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)

    return app
