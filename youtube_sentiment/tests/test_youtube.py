from unittest import TestCase
import requests
import json
import shutil
from youtube_sentiment import Youtube

class TestYT(TestCase):
    """ Test Youtube """
    @classmethod
    def setUpClass(self):
        """ Setup """
        self.yt = Youtube('FAKE_API_KEY', 'FAKE_VIDEO_ID', 9999) # for testing only, service creation

    """ Youtube API test cases """
    def test_endpoint(self):
        """ Check if endpoint is up """
        r = requests.get(url='https://www.googleapis.com/youtube/v3/commentThreads')
        self.assertTrue(r.status_code == requests.codes.ok or 
                r.status_code == requests.codes.bad_request)

    def test_comments_structure(self):
        """ Check Youtube class comments structure for json """
        param = json.loads(
                r"""{"kind":"youtube#commentThreadListResponse","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/R0FUhvhNOTLQstdhxRu2bjXjF30\"","nextPageToken":"QURTSl9pMmM2dGJzVWN3THJKSTVSVlVIOVMyb0pCc2l0XzFTenNKc2NsX3NRaGJpUDkzZEZlNHAyQnhVYlE4RW1GMjhXUHNLUm0xRnYtckFlaWlrNkxMTm1DR1RVN0IxRHJPMk0zLVJJLVBCYml6T3Y1aVdWUWEzTzlDbElwWkZFdkk=","pageInfo":{"totalResults":20,"resultsPerPage":20},"items":[{"kind":"youtube#commentThread","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/2VfXK_dU5DGJ0xO0zDjAcg1McHE\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"videoId":"J0zySGE82HY","topLevelComment":{"kind":"youtube#comment","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/TP_MDn7uTvkmpr16XzTjqleaEmU\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"authorDisplayName":"Extra Neatro","authorProfileImageUrl":"https://yt3.ggpht.com/-2GG1k5EOKW8/AAAAAAAAAAI/AAAAAAAAAAA/3gl5ZgLC59U/s28-c-k-no-mo-rj-c0xffffff/photo.jpg","authorChannelUrl":"http://www.youtube.com/channel/UCxHIlxtQQkaOey8SDnMaGsA","authorChannelId":{"value":"UCxHIlxtQQkaOey8SDnMaGsA"},"videoId":"J0zySGE82HY","textDisplay":"Steak","textOriginal":"\"Steak\"","canRate":true,"viewerRating":"none","likeCount":0,"publishedAt":"2018-10-15T18:43:46.000Z","updatedAt":"2018-10-15T18:43:46.000Z"}},"canReply":true,"totalReplyCount":0,"isPublic":true}}]}"""
            )
        result = self.yt.get_comments_values(param)
        self.assertEqual(result, ["Steak"])

    @classmethod
    def tearDownClass(cls):
        """ Tear down """
        shutil.rmtree('./log')