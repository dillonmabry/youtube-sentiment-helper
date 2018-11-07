from unittest import TestCase
import shutil
import os
from youtube_sentiment import Service
from requests import codes

class TestService(TestCase):
    """ Test Service layer """

    def test_service_attributes(self):
        """ Ensure service attributes """
        with Service('') as mock_service:
            self.assertTrue(hasattr(mock_service, 'get'))
            self.assertTrue(hasattr(mock_service, '__enter__'))
            self.assertTrue(hasattr(mock_service, '__exit__'))
            self.assertTrue(type(mock_service.__enter__()) is type(mock_service))

    """ Service test cases """
    def test_ok_endpoint(self):
        """ Test sample endpoint """
        with Service('https://www.google.com') as mock_service:
            payload = {
                'q': 'test'
            }
            r = mock_service.get("https://www.google.com/search")
            self.assertTrue(r.status_code == codes.ok)
    
    def test_request_exception(self):
        """ Test sample endpoint """
        with self.assertRaises(Exception) as mockexception:
            with Service('https://www.googleapis.com/youtube/v3/commentThreads') as mock_service:
                payload = {
                    'key': 'FAKE_API_KEY', 
                    'textFormat': 'plaintext', 
                    'part': 'snippet', 
                    'videoId': '9999',
                    'maxResults': 100
                }
                r = mock_service.get(payload=payload)