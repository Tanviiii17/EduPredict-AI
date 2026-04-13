import os
import json
import logging
from datetime import datetime
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

MODEL_PATH = 'models/adaptability_model.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
METRICS_PATH = 'models/metrics.json'

FEATURE_ORDER = [
    'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
    'Location', 'Load-shedding', 'Financial Condition', 'Internet Type',
    'Network Type', 'Class Duration', 'Self Lms', 'Device'
]

class MLService:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.metrics = {}
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
                logger.warning("Model not found. Attempting to train...")
                from scripts.training import model_training
                model_training.train_model()

            self.model = joblib.load(MODEL_PATH)
            self.encoders = joblib.load(ENCODER_PATH)

            with open(METRICS_PATH, 'r') as f:
                self.metrics = json.load(f)

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_single(self, data):
        features = []
        for feature in FEATURE_ORDER:
            if feature not in data:
                raise ValueError(f'Missing feature: {feature}')

            value = data[feature]
            if feature in self.encoders:
                if value not in self.encoders[feature].classes_:
                    raise ValueError(f'Invalid value for {feature}: {value}')
                encoded_value = self.encoders[feature].transform([value])[0]
            else:
                encoded_value = value
            features.append(encoded_value)

        features_array = pd.DataFrame([features], columns=FEATURE_ORDER)

        prediction = self.model.predict(features_array)[0]
        probabilities = self.model.predict_proba(features_array)[0]

        predicted_label = self.encoders['Adaptivity Level'].inverse_transform([prediction])[0]
        class_labels = self.encoders['Adaptivity Level'].classes_
        prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}

        confidence = float(max(probabilities) * 100)

        # Approximate feature importance for Explainability (Mock logic for GNB)
        # We will look at probability differences or just return a mock importance for now
        # until we can compute it rigorously
        feature_importance = [
            {"feature": "Financial Condition", "impact": "negative", "score": -0.15},
            {"feature": "Load-shedding", "impact": "negative", "score": -0.12},
            {"feature": "Institution Type", "impact": "positive", "score": 0.08},
        ]

        result = {
            'prediction': predicted_label,
            'confidence': round(confidence, 2),
            'probabilities': {k: round(v * 100, 2) for k, v in prob_dict.items()},
            'explainability': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        return result

    def predict_batch(self, df):
        for feature in FEATURE_ORDER:
            if feature not in df.columns:
                raise ValueError(f'Missing column: {feature}')

        encoded_df = df.copy()
        for feature in FEATURE_ORDER:
            if feature in self.encoders:
                try:
                    encoded_df[feature] = self.encoders[feature].transform(df[feature])
                except ValueError as e:
                    raise ValueError(f'Invalid values in column {feature}')

        X = encoded_df[FEATURE_ORDER]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        predicted_labels = self.encoders['Adaptivity Level'].inverse_transform(predictions)

        df['Predicted_Adaptivity'] = predicted_labels
        df['Confidence'] = [round(max(prob) * 100, 2) for prob in probabilities]
        return df

    def get_metrics(self):
        return self.metrics

    def retrain(self):
        from scripts.training import model_training
        model_training.train_model()
        self.load_model()
        return self.metrics

ml_service = MLService()
