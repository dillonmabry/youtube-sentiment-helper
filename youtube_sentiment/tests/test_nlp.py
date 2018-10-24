from unittest import TestCase
from youtube_sentiment import NLP

class TestUtil(TestCase):
    """ Test Utility """
    @classmethod
    def setUpClass(self):
        """ Setup """
        self.mock = NLP("lr_sentiment_basic.pkl")

    """ Utility class tests """
    def test_tagged_comments(self):
        """ Test tagged video comments """
        mock_comments = ["Was a great movie", "Terrible movie and awful not recommend"]
        tagged = self.mock.process_comments(mock_comments)
        self.assertTrue(type(tagged) == list)
        self.assertTrue(tagged[0][1] == 1)
        self.assertTrue(tagged[1][1] == 0)