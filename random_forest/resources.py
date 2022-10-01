from flask import request
from flask_restx import Resource

from random_forest.docs import predict_namespace, predict_request, predict_response
from random_forest.services import predict, get_model


@predict_namespace.route('/')
@predict_namespace.doc()
class PredictResource(Resource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @predict_namespace.expect(predict_request)
    @predict_namespace.response(code=400, description='Bad Request')
    @predict_namespace.response(code=201, description='Success', model=predict_response)
    def post(self):
        data = request.get_json()
        model = get_model(data)
        response = predict(model)
        return response, 201
