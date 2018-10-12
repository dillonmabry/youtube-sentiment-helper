import requests
from requests.exceptions import RequestException
from logger import Logger

class Youtube(object):
    """
    Main class to use REST requests
    Args:
        self
        endpoint: Google API endpoint
        api_key: Google API key for Youtube
    """
    def __init__(self, endpoint, api_key):
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.logger = Logger(self.__class__.__name__).get()

    def get(self, videoId):
        """
        Method to return video comments
        Args:
            self
            videoId: Youtube video unique id from url
        """
        try:
            payload = {
                'key': self.api_key, 
                'textFormat': 'plaintext', 
                'part': 'snippet', 
                'videoId': videoId,
                'maxResults': 100
            }
            r = self.session.request(method='get', url=self.endpoint, params=payload)
            if (r.status_code == requests.codes.ok):
                self.logger.info("API request endpoint: {0} | Video requested: {1}".format(self.endpoint, videoId))
            return r.json()
        except RequestException as e:  # This is the correct syntax
            self.logger.exception(str(e))
            raise