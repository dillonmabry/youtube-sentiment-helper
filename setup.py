from setuptools import setup

setup(name='youtube_sentiment',
      version='0.1',
      description='Analyze Youtube videos for general sentiment analysis',
      url='https://github.com/dillonmabry/youtube-sentiment-helper',
      author='Dillon Mabry',
      author_email='rapid.dev.solutions@gmail.com',
      license='MIT',
      packages=['youtube_sentiment'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      data_files=[('', [
            'youtube_sentiment/models/lr_sentiment_basic.pkl', 
            'youtube_sentiment/models/lr_sentiment_cv.pkl'])
      ],
      zip_safe=False)