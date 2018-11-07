import requests
from requests.exceptions import RequestException
from youtube_sentiment import Logger

class Service(object):
    """
    Service helper class for API requests
    Args:
        endpoint: URL endpoint
    """
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.logger = Logger(self.__class__.__name__, maxbytes=10*1024*1024).get()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def get(self, payload):
        try:
            r = self.session.request(method='get', url=self.endpoint, params=payload)
            if (r.status_code == requests.codes.ok):
                self.logger.info("API endpoint request received, payload: {0}".format(payload))
                return r
            elif (r.status_code == requests.codes.bad_request):
                self.logger.error("Status: {0} | Bad Request. Check API Key or connectivity".format(r.status_code))
                raise Exception("Bad Request")
            elif (r.status_code == requests.codes.forbidden):
                self.logger.error("Status: {0} | API Key is incorrect or restricted".format(r.status_code))
                raise Exception("Forbidden")
            elif (r.status_code == requests.codes.not_found): 
                self.logger.error("Status: {0} | Video not found".format(r.status_code))
                raise Exception("Not Found")
            else:
                self.logger.error("Status: {0} | An error has occurred".format(r.status_code))
                raise Exception("Internal server error")
        except RequestException as e:
            self.logger.exception(str(e))
            raise
            
