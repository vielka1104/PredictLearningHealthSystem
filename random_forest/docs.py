from flask_cors import cross_origin
from flask_restx import Namespace, fields

predict_namespace = Namespace('predict', description='Predict Operations', decorators=[cross_origin()])

predict_request = predict_namespace.model('PredictRequest', {
    'is_contact_with_infected': fields.Boolean(required=True),
    'observation': fields.String(required=True),
    'symptoms': fields.List(fields.Integer)
})

predict_response = predict_namespace.model('PredictResponse', {
    'survive': fields.Integer,
    'percentage': fields.Integer,
})

