from unittest import TestCase
import shutil
from youtube_sentiment import Logger

class TestLogger(TestCase):
    """ Test logger """
    @classmethod
    def setUpClass(self):
        """ Setup """
        self.mock = Logger(self.__class__.__name__, maxbytes=8)

    """ Logging tests """
    def test_logging(self):
        """ Test logger init and log creation """
        self.assertTrue(hasattr(self.mock, 'get'))
        logger = self.mock.get()
        self.assertTrue(hasattr(logger, 'info'))
        logger.info("log test")

    @classmethod
    def tearDownClass(self):
        """ Tear down """
        shutil.rmtree('./log')
