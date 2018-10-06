Naive Bayes spamfilter implementation in python 3.6 and its command line interface for Linux system

Main files:

spamfilter.py
NBclassifier.py

Usage:

  spamfilter train    -m <path> -c <path>
  spamfilter classify -m <path> <path>

Options:

  -c <path>, --corpuspath  Path to the corpus that should be used to train the
                           spamfilter. The path should contain two
                           directories, ham and spam, which contain the ham
                           and spam emails in individual files.
  -m <path>, --modelpath   Path to the model. In training mode this is where
                           the model will be written to, in classification
                           mode this is where the model will be read from.
