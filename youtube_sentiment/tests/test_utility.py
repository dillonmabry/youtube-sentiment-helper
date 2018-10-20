from unittest import TestCase
from youtube_sentiment import flatten_list
from youtube_sentiment import load_ml_pipeline
from youtube_sentiment import top_freq_words

class TestUtil(TestCase):
    """ Test Utility """
    @classmethod
    def setUpClass(self):
        """ Setup """
        self.mock = load_ml_pipeline("lr_sentiment_basic.pkl")

    """ Utility class tests """
    def test_flatten_list(self):
        """ Test flatten list of lists structure """
        mock = [["Was a great movie"], ["Wow did you see that?"]]
        self.assertTrue(flatten_list(mock) == ["Was a great movie", "Wow did you see that?"])

    def test_load_model(self):
        """ Test loading of a model """
        self.assertTrue(self.mock != None)
        self.assertTrue(hasattr(self.mock, 'predict'))

    def test_model_predict(self):
        """ Test model sentiment predict and action """
        mock_comments = ["Hey nice video you made loved it", "Terrible video worst ever"]
        predictions = self.mock.predict(mock_comments)
        self.assertTrue(predictions[0] == 1)
        self.assertTrue(predictions[1] == 0)

    def test_comments_frequency(self):
        """ Test comments word frequencies """
        mock_comments = "Hey nice video you made loved it Terrible video worst ever loved it loved"
        top_words = top_freq_words(mock_comments, topwords=20)
        self.assertTrue(len(top_words) > 1)
        print(top_words[0])
        self.assertTrue(top_words[0] == ("loved", 3))