import argparse
from youtube import Youtube
import time

def process_video_comments(apiKey, videoId, maxpages):
    yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', apiKey, maxpages)
    start_time = time.time()
    comments = yt.get_comments(videoId)
    print(len(comments))
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("apiKey", help="Enter the Youtube API key to use for requests")
    parser.add_argument("videoId", help="Enter the Youtube video ID")
    parser.add_argument("maxpages", help="Enter the max pages returned of comments", type=int)
    args = parser.parse_args()
    process_video_comments(args.apiKey, args.videoId, args.maxpages)