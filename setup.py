from setuptools import setup, find_packages

setup(name='youtube_sentiment',
      version='0.1',
      description='Analyze Youtube videos for general sentiment analysis',
      url='https://github.com/dillonmabry/youtube-sentiment-helper',
      author='Dillon Mabry',
      author_email='rapid.dev.solutions@gmail.com',
      license='MIT',
      packages=find_packages('youtube_sentiment'),
      entry_points={
      'console_scripts': [
            'youtube_sentiment=youtube_sentiment.main:main',
      ],
      },
      zip_safe=False)