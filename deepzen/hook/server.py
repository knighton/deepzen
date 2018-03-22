from flask import Flask, request
import logging
import requests
from threading import Thread

from .base.hook import Hook


class Server(Hook):
    def __init__(self, host='0.0.0.0', port=31337):
        self.host = host
        self.port = port

    def run(self):
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        app = Flask(__name__)

        @app.route('/')
        def serve_root():
            return open('deepzen/hook/server/index.html').read()

        @app.route('/stop', methods=['POST'])
        def server_stop():
            func = request.environ.get('werkzeug.server.shutdown')
            if func is not None:
                func()
            return ''

        print('Model training web interface: http://%s:%d/' %
              (self.host, self.port))
        app.run(host=self.host, port=self.port)

    def on_fit_begin(self, metric_name_lists, epoch_offset, epochs):
        self.metric_name_lists = metric_name_lists
        self.epoch_offset = epoch_offset
        self.epochs = epochs
        self.thread = Thread(target=self.run)
        self.thread.start()

    def stop_server(self):
        requests.post('http://%s:%d/stop' % (self.host, self.port))

    def on_fit_end(self):
        self.stop_server()
