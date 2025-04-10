from flask import Flask
from pymilvus import connections
from config import Config


def create_app(config_class=Config):
    # Initialize Flask application
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Connect to Milvus
    try:
        connections.connect(
            host=app.config["MILVUS_HOST"], port=app.config["MILVUS_PORT"]
        )
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")

    # Register blueprints
    from app.routes import main

    app.register_blueprint(main)

    return app
