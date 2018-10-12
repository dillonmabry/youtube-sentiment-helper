import config
from youtube import Youtube

if __name__ == '__main__':
    yt = Youtube(config.YOUTUBE_ENDPOINT, config.YOUTUBE_API_KEY)
    comments = yt.get_comments('0Z3ZjnwbA3s')
    print(comments)