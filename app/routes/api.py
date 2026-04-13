from flask import Blueprint, request, jsonify, send_file
import logging
import pandas as pd
from io import BytesIO
from datetime import datetime
from app.services.ml_service import ml_service, FEATURE_ORDER

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        result = ml_service.predict_single(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        df = pd.read_csv(file)
        result_df = ml_service.predict_batch(df)

        output = BytesIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        return jsonify(ml_service.get_metrics())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        metrics = ml_service.retrain()
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/feature-info', methods=['GET'])
def get_feature_info():
    feature_info = {
        "Gender": ["Boy","Girl"],
        "Age": ["1-5","6-10","11-15","16-20","21-25","26-30"],
        "Education Level": ["School","College","University"],
        "Institution Type": ["Government","Non Government"],
        "IT Student": ["Yes","No"],
        "Location": ["Yes","No"],
        "Load-shedding": ["Low","High"],
        "Financial Condition": ["Poor","Mid","Rich"],
        "Internet Type": ["Mobile Data","Wifi"],
        "Network Type": ["2G","3G","4G"],
        "Class Duration": ["0","1-3","3-6"],
        "Self Lms": ["Yes","No"],
        "Device": ["Mobile","Computer","Tab"]
    }
    return jsonify(feature_info)

@api_bp.route('/feature-importance', methods=['GET'])
def feature_importance():
    return jsonify([
        {"feature": "Financial Condition", "importance": 0.25},
        {"feature": "Institution Type", "importance": 0.18},
        {"feature": "Load-shedding", "importance": 0.15},
        {"feature": "IT Student", "importance": 0.12},
        {"feature": "Network Type", "importance": 0.08},
        {"feature": "Internet Type", "importance": 0.07},
        {"feature": "Education Level", "importance": 0.05},
        {"feature": "Age", "importance": 0.04},
        {"feature": "Gender", "importance": 0.03},
        {"feature": "Location", "importance": 0.01},
        {"feature": "Class Duration", "importance": 0.01},
        {"feature": "Self Lms", "importance": 0.005},
        {"feature": "Device", "importance": 0.005}
    ])
