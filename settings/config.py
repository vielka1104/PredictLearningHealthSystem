class Config(object):
    DEBUG = True
    TESTING = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    PROPAGATE_EXCEPTIONS = True
    SECRET_KEY = 'super secret'
    SCHEDULER_API_ENABLED = True


class DevelopmentConfig(Config):
    PROPAGATE_EXCEPTIONS = True
    SECRET_KEY = 'super secret'
    SCHEDULER_API_ENABLED = True
