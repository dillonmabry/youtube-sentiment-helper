from setuptools import setup
from setuptools.command.install import install

def readme():
      with open('README.md') as f:
            return f.read()

def post_install():
      """Post installation nltk corpus downloads."""
      import nltk
      nltk.download("punkt")
      nltk.download('words')
      nltk.download('maxent_ne_chunker')
      nltk.download('averaged_perceptron_tagger')
      nltk.download("stopwords")

class PostInstall(install):
      """Post-installation"""
      def run(self):
            install.run(self)
            self.execute(post_install, [], msg="Running post installation tasks")

setup(name='youtube_sentiment',
      version='0.2.1',
      description='Analyze Youtube videos for general sentiment analysis',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/dillonmabry/youtube-sentiment-helper',
      author='Dillon Mabry',
      author_email='rapid.dev.solutions@gmail.com',
      license='MIT',
      packages=['youtube_sentiment'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=["requests", "nltk", "numpy", "scipy", "scikit-learn"],
      cmdclass={"install": PostInstall},
      include_package_data=True,
      data_files=[('', [
            'youtube_sentiment/models/lr_sentiment_basic.pkl', 
            'youtube_sentiment/models/lr_sentiment_cv.pkl'])
      ],
      zip_safe=False)