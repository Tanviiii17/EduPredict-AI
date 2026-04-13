from flask import Blueprint, render_template

views_bp = Blueprint('views', __name__)

@views_bp.route('/')
def index():
    """Serve the main HTML dashboard"""
    return render_template('index.html')
