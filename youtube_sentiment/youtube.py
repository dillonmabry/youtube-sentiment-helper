from youtube_sentiment import Service
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
        self.apiService = Service(endpoint)
        self.api_key = api_key
        self.maxpages = maxpages

    def get_comments(self, videoId):
        """
        Method to return list of video comments
        Args:
            self
            videoId: Youtube video unique id from url
        """
        payload = {
            'key': self.api_key, 
            'textFormat': 'plaintext', 
            'part': 'snippet', 
            'videoId': videoId,
            'maxResults': 100
        }
        r = self.apiService.get(payload=payload)
        all_comments = []
        all_comments.append(self.get_comments_threads(r.json()))
        nextPageToken = r.json().get('nextPageToken')
        idx = 0
        while(nextPageToken and idx < self.maxpages):
            payload["pageToken"] = nextPageToken
            r_next = self.apiService.get(payload=payload)
            nextPageToken = r_next.json().get("nextPageToken")
            all_comments.append(self.get_comments_threads(r_next.json()))
            idx += 1
        return flatten_list(all_comments)

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
        except KeyError as keyError:
            raise
        except Exception as e:
            raise
