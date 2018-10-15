# Youtube Sentiment Helper
Determine sentiment of Youtube video per comment based analysis using Sci-kit by analyzing video comments based on positive/negative sentiment. 
Helper tool to make requests to a machine learning model in order to determine sentiment using the Youtube API.

## Install Instructions
`pip install .`
## How to Use
Current usage:
```
python main.py <Youtube API Key> <Youtube video ID> <Max Pages of Comments>
```
or
```
import youtube_sentiment as yt
yt.process_video_comments(<Youtube API Key>, <Youtube video ID>, <Max Pages of Comments>) 
```
## To-Do
- [X] Create API to use Youtube API V3 via REST to get comments for videos
- [X] Analyze existing sentiment analysis models to select and use
- [ ] Improve/enhance sentiment learning model
- [ ] Utilize sentiment analysis to analyze Youtube video and provide analytics
- [ ] Finalize and create Python package for project 
- [ ] Create web based portal
