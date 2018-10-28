import requests
from requests.exceptions import RequestException
from youtube_sentiment import Logger
from youtube_sentiment import flatten_list

class Youtube(object):
    """
    Main class to use REST requests using Google Youtube API V3
    https://developers.google.com/youtube/v3/docs/
    Args:
        self
        endpoint: Google API endpoint
        api_key: Google API key for Youtube
    """
    def __init__(self, endpoint, api_key, maxpages):
        self.endpoint = endpoint
        self.api_key = api_key
        self.maxpages = maxpages
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        self.logger = Logger(self.__class__.__name__, maxbytes=10*1024*1024).get()

    def get_comments(self, videoId):
        """
        Method to return list of video comments
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
                all_comments = []
                self.logger.info("API request endpoint: {0} | Video requested: {1}".format(
                    self.endpoint, videoId))
                all_comments.append(self.get_comments_threads(r.json()))
                nextPageToken = r.json().get('nextPageToken')
                idx = 0
                while(nextPageToken and idx < self.maxpages):
                    payload["pageToken"] = nextPageToken
                    r_next = self.session.request(method='get', url=self.endpoint, params=payload)
                    if(r_next.status_code == requests.codes.ok):
                        nextPageToken = r_next.json().get("nextPageToken")
                        all_comments.append(self.get_comments_threads(r_next.json()))
                        idx = idx + 1
                return flatten_list(all_comments)
            elif (r.status_code == requests.codes.forbidden):
                self.logger.error("Status: {0} | API Key is incorrect or restricted.".format(r.status_code))
                raise Exception("API Key is incorrect or not valid")
            elif (r.status_code == requests.codes.not_found): 
                self.logger.error("Status: {0} | Video not found".format(r.status_code))
                raise Exception("VideoId not found or internal Youtube API error")
            else:
                self.logger.error("Status: {0} | An error has occurred".format(r.status_code))
                raise Exception("API Key is incorrect or internal Youtube API error")
        except RequestException as e:
            self.logger.exception(str(e))
            raise

    def get_comments_threads(self, comments):
        """
        Method to return all comments from Youtube comment threads
        Args:
            self
            comments: list of response json comments including replies
        """
        try:
            all_comments = []
            for item in comments["items"]:
                comment = item["snippet"]["topLevelComment"]
                text = comment["snippet"]["textDisplay"]
                all_comments.append(text)
                if 'replies' in item.keys():
                    for reply in item['replies']['comments']:
                        rtext = reply["snippet"]["textDisplay"]
                        all_comments.append(rtext)
            return all_comments
        except Exception as e:
            self.logger.exception(str(e))
            raise
