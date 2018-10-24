import argparse
from youtube_sentiment import Youtube
from youtube_sentiment import NLP

def video_summary(apiKey, videoId, maxpages, model):
    yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', apiKey, maxpages)
    comments = yt.get_comments(videoId)
    nlp = NLP(model)
    nlp.process_comments_summary(comments)

def tagged_comments(apiKey, videoId, maxpages, model):
    yt = Youtube('https://www.googleapis.com/youtube/v3/commentThreads', apiKey, maxpages)
    comments = yt.get_comments(videoId)
    nlp = NLP(model)
    tagged_comments = nlp.process_comments(comments)
    return tagged_comments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("apiKey", help="Enter the Youtube API key to use for requests")
    parser.add_argument("videoId", help="Enter the Youtube video ID")
    parser.add_argument("maxpages", help="Enter the max pages returned of comments", type=int)
    parser.add_argument("model", help="Enter the model name to use for sentiment")
    args = parser.parse_args()
    video_summary(args.apiKey, args.videoId, args.maxpages, args.model)

if __name__ == '__main__':
    main()