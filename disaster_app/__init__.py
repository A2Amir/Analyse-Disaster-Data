from flask import Flask

app = Flask(__name__)

from disaster_app import routes
