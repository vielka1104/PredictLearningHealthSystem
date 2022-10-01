from flask import Blueprint
from flask_restx import Api
from random_forest.resources import predict_namespace
predict_blueprint = Blueprint('predict_api', __name__, url_prefix='/api')
api = Api(predict_blueprint, title='Predict API', description='A predict api', doc='/doc')

api.add_namespace(predict_namespace)
