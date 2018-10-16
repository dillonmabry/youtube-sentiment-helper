from unittest import TestCase
import json
from youtube_sentiment import flatten_list

class TestUtil(TestCase):
    """ Utility class tests """
    def test_flatten_list(self):
        """ Test flatten list of lists structure """
        mock = [["Was a great movie"], ["Wow did you see that?"]]
        self.assertTrue(flatten_list(mock) == ["Was a great movie", "Wow did you see that?"])