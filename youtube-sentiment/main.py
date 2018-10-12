import config
from youtube import Youtube
from utility import get_key

if __name__ == '__main__':
    yt = Youtube(config.YOUTUBE_ENDPOINT, config.YOUTUBE_API_KEY)
    res = yt.get('0Z3ZjnwbA3s')
    comments = get_key(res, "textDisplay")
    print(comments[0])