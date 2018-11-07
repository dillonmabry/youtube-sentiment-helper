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
        self.yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', 'MOCK_API_KEY', 5) # for testing only, service creation

    """ Youtube API test cases """
    def test_endpoint(self):
        """ Check if endpoint is up """
        r = requests.get(url='https://www.googleapis.com/youtube/v3/commentThreads')
        self.assertTrue(r.status_code == requests.codes.ok or 
                r.status_code == requests.codes.bad_request)

    def test_api_errors(self):
        """ Test requests errors handling """
        with self.assertRaises(Exception) as mock:
            r = self.yt.get_comments("wyG2xN5")

    def test_comments_threads(self):
        """ Check Youtube class comments structure for json """
        param = json.loads(
                r"""{"kind":"youtube#commentThreadListResponse","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/R0FUhvhNOTLQstdhxRu2bjXjF30\"","nextPageToken":"QURTSl9pMmM2dGJzVWN3THJKSTVSVlVIOVMyb0pCc2l0XzFTenNKc2NsX3NRaGJpUDkzZEZlNHAyQnhVYlE4RW1GMjhXUHNLUm0xRnYtckFlaWlrNkxMTm1DR1RVN0IxRHJPMk0zLVJJLVBCYml6T3Y1aVdWUWEzTzlDbElwWkZFdkk=","pageInfo":{"totalResults":20,"resultsPerPage":20},"items":[{"kind":"youtube#commentThread","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/2VfXK_dU5DGJ0xO0zDjAcg1McHE\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"videoId":"J0zySGE82HY","topLevelComment":{"kind":"youtube#comment","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/TP_MDn7uTvkmpr16XzTjqleaEmU\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"authorDisplayName":"Extra Neatro","authorProfileImageUrl":"https://yt3.ggpht.com/-2GG1k5EOKW8/AAAAAAAAAAI/AAAAAAAAAAA/3gl5ZgLC59U/s28-c-k-no-mo-rj-c0xffffff/photo.jpg","authorChannelUrl":"http://www.youtube.com/channel/UCxHIlxtQQkaOey8SDnMaGsA","authorChannelId":{"value":"UCxHIlxtQQkaOey8SDnMaGsA"},"videoId":"J0zySGE82HY","textDisplay":"Steak","textOriginal":"\"Steak\"","canRate":true,"viewerRating":"none","likeCount":0,"publishedAt":"2018-10-15T18:43:46.000Z","updatedAt":"2018-10-15T18:43:46.000Z"}},"canReply":true,"totalReplyCount":0,"isPublic":true}}]}"""
            )
        result = self.yt.get_comments_threads(param)
        self.assertEqual(result, ["Steak"])

    def test_comments_replies(self):
        """ Check Youtube class replies structure for json """
        param = json.loads(
                r"""{"kind":"youtube#commentThreadListResponse","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/R0FUhvhNOTLQstdhxRu2bjXjF30\"","nextPageToken":"QURTSl9pMmM2dGJzVWN3THJKSTVSVlVIOVMyb0pCc2l0XzFTenNKc2NsX3NRaGJpUDkzZEZlNHAyQnhVYlE4RW1GMjhXUHNLUm0xRnYtckFlaWlrNkxMTm1DR1RVN0IxRHJPMk0zLVJJLVBCYml6T3Y1aVdWUWEzTzlDbElwWkZFdkk=","pageInfo":{"totalResults":20,"resultsPerPage":20},"items":[{"kind":"youtube#commentThread","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/2VfXK_dU5DGJ0xO0zDjAcg1McHE\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"videoId":"J0zySGE82HY","topLevelComment":{"kind":"youtube#comment","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/TP_MDn7uTvkmpr16XzTjqleaEmU\"","id":"Ugy5WCt4bMYfI6MXJGh4AaABAg","snippet":{"authorDisplayName":"Extra Neatro","authorProfileImageUrl":"https://yt3.ggpht.com/-2GG1k5EOKW8/AAAAAAAAAAI/AAAAAAAAAAA/3gl5ZgLC59U/s28-c-k-no-mo-rj-c0xffffff/photo.jpg","authorChannelUrl":"http://www.youtube.com/channel/UCxHIlxtQQkaOey8SDnMaGsA","authorChannelId":{"value":"UCxHIlxtQQkaOey8SDnMaGsA"},"videoId":"J0zySGE82HY","textDisplay":"Steak","textOriginal":"\"Steak\"","canRate":true,"viewerRating":"none","likeCount":0,"publishedAt":"2018-10-15T18:43:46.000Z","updatedAt":"2018-10-15T18:43:46.000Z"}},"canReply":true,"totalReplyCount":0,"isPublic":true},"replies":{"comments":[{"kind":"youtube#comment","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/gAj-IQaO8qbIvHqB8rRAGAxeOME\"","id":"UgzWYfj2xGWv6a9nIT54AaABAg.8m6qqHgWu5q8mKJmcw2DrX","snippet":{"authorDisplayName":"Mickael Bergeron Neron","authorProfileImageUrl":"https://yt3.ggpht.com/-zfuCxyb0_-I/AAAAAAAAAAI/AAAAAAAAAAA/1Prt6wek_oQ/s28-c-k-no-mo-rj-c0xffffff/photo.jpg","authorChannelUrl":"http://www.youtube.com/channel/UCFI9lOY7S_Xc2bzINia8tZA","authorChannelId":{"value":"UCFI9lOY7S_Xc2bzINia8tZA"},"textDisplay":"Awesome video","textOriginal":"Awesome video","parentId":"UgzWYfj2xGWv6a9nIT54AaABAg","canRate":true,"viewerRating":"none","likeCount":1,"publishedAt":"2018-10-13T02:24:39.000Z","updatedAt":"2018-10-13T02:24:58.000Z"}},{"kind":"youtube#comment","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/gAj-IQaO8qbIvHqB8rRAGAxeOME\"","id":"UgzWYfj2xGWv6a9nIT54AaABAg.8m6qqHgWu5q8mKJmcw2DrX","snippet":{"authorDisplayName":"Mickael Bergeron Neron","authorProfileImageUrl":"https://yt3.ggpht.com/-zfuCxyb0_-I/AAAAAAAAAAI/AAAAAAAAAAA/1Prt6wek_oQ/s28-c-k-no-mo-rj-c0xffffff/photo.jpg","authorChannelUrl":"http://www.youtube.com/channel/UCFI9lOY7S_Xc2bzINia8tZA","authorChannelId":{"value":"UCFI9lOY7S_Xc2bzINia8tZA"},"textDisplay":"Great Stuff","textOriginal":"Great Stuff","parentId":"UgzWYfj2xGWv6a9nIT54AaABAg","canRate":true,"viewerRating":"none","likeCount":2,"publishedAt":"2018-10-13T02:24:39.000Z","updatedAt":"2018-10-13T02:24:58.000Z"}}]}}]}"""
            )
        result = self.yt.get_comments_threads(param)
        self.assertTrue("Great Stuff" in result)
        self.assertTrue("Awesome video" in result)

    def test_comments_validation(self):
        """ Check Youtube class comments structure for json, key validation """
        param = json.loads(
                r"""{"kind":"youtube#commentThreadListResponse","etag":"\"XI7nbFXulYBIpL0ayR_gDh3eu1k/R0FUhvhNOTLQstdhxRu2bjXjF30\"","nextPageToken":"QURTSl9pMmM2dGJzVWN3THJKSTVSVlVIOVMyb0pCc2l0XzFTenNKc2NsX3NRaGJpUDkzZEZlNHAyQnhVYlE4RW1GMjhXUHNLUm0xRnYtckFlaWlrNkxMTm1DR1RVN0IxRHJPMk0zLVJJLVBCYml6T3Y1aVdWUWEzTzlDbElwWkZFdkk=","pageInfo":{"totalResults":20,"resultsPerPage":20}}"""
            )
        with self.assertRaises(KeyError) as mockexception:
            result = self.yt.get_comments_threads(param)

    @classmethod
    def tearDownClass(cls):
        """ Tear down """
        shutil.rmtree('./log')