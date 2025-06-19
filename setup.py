from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
  name = 'trainer_pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Trainer Pytorch',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Francisco Carrillo Pérez',
  author_email = 'carrilloperezfrancisco@gmail.com',
  url = 'https://github.com/pacocp/trainer_pytorch',
  keywords = [
    'artificial intelligence',
    'pytorch',
    'machine learning',
  ],
  install_requires=[
    'wandb>=0.20.1',
    'accelerate>=0.10.0',
    'torch>=2.2',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==2.2',
    'wandb==0.20.1',
    'accelerate==0.10.0'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)