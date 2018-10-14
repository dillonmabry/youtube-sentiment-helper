import config
from youtube import Youtube
import time

if __name__ == '__main__':
    yt = Youtube(config.YOUTUBE_ENDPOINT, config.YOUTUBE_API_KEY)
    start_time = time.time()
    comments = yt.get_comments('0Z3ZjnwbA3s')
    print(comments)
    print("--- %s seconds ---" % (time.time() - start_time))