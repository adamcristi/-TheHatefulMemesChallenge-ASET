# -*- coding: utf-8 -*-
import unittest
import logging.handlers

class MockLoggingHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.reset()
        super(MockLoggingHandler, self).__init__(*args, **kwargs)

    def emit(self, record):
        self.messages.append(record)

    def reset(self):
        self.messages = []

class TestLogging(unittest.TestCase):
    def setUp(self):
        self.logging_handler = MockLoggingHandler()
        logging.getLogger('pythonrv_python3').addHandler(self.logging_handler)

    def tearDown(self):
        logging.getLogger('pythonrv_python3').removeHandler(self.logging_handler)

    def assertLog(self, level, msg):
        record = self.logging_handler.messages[-1]
        self.assertEqual(record.levelno, level)
        self.assertEqual(record.getMessage(), msg)
