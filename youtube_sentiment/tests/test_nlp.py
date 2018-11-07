from unittest import TestCase
import sys
from contextlib import contextmanager
from io import StringIO
from youtube_sentiment import NLP

class TestUtil(TestCase):
    """ Test Utility """
    @classmethod
    def setUpClass(self):
        """ Setup """
        self.mock = NLP("lr_sentiment_basic.pkl")

    @contextmanager
    def captured_output(self):
        """ To capture stdout or errors """
        new_out, new_err = StringIO(), StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    """ Utility class tests """
    def test_process_comments(self):
        """ Test tagged video comments """
        mock_comments = ["Was a great movie", "Terrible movie and awful not recommend"]
        tagged = self.mock.process_comments(mock_comments)
        self.assertTrue(type(tagged) == list)
        self.assertTrue(tagged[0][0] == "Was a great movie")
        self.assertTrue(tagged[1][0] == "Terrible movie and awful not recommend")
        self.assertTrue(tagged[0][1] == 1)
        self.assertTrue(tagged[1][1] == 0)

    def test_comments_summary(self):
        """ Test NLP summary """
        mock_comments = ["Was a great movie the character Jude was good", "Terrible movie and awful not recommend", "Worst I have ever seen"]
        with self.captured_output() as (out, err):
            self.mock.process_comments_summary(mock_comments)
            printout = out.getvalue().strip() # capture stdout for summary test
            self.assertTrue("Total sentiment scores (Pos, Neg): 1, 2" in printout)
            self.assertTrue("Percentage Negative Sentiment: 0.6666666666666666" in printout)
            self.assertTrue("Top words by frequency: [('movie', 2)" in printout)
            self.assertTrue("Key entities: ['Jude']" in printout)