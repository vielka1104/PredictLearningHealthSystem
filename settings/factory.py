from flask import Flask
from settings.config import ProductionConfig
from settings.exceptions import NotFoundException, BadRequestException, EmailException, InternalServerException,\
    handle_exception
from settings.layers.serialization import ma
from random_forest.urls import predict_blueprint
from random_forest.services import classifier


def create_app():
    app = Flask(__name__)
    config = ProductionConfig()
    app.config.from_object(config)
    ma.init_app(app)
    # with app.app_context():
    #     db.create_all()
    app.app_context().push()
    app.register_blueprint(predict_blueprint)
    app.register_error_handler(BadRequestException, handle_exception)
    app.register_error_handler(NotFoundException, handle_exception)
    app.register_error_handler(EmailException, handle_exception)
    app.register_error_handler(InternalServerException, handle_exception)
    classifier()
    return app
