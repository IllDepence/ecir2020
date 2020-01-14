import re
from gensim.parsing.preprocessing import (strip_multiple_whitespaces,
                                          strip_punctuation,
                                          remove_stopwords)

def bow_preprocess_string(s):
    """ Preprocessing of strings to be used for gensim dictionary generation
        and for turning test set strings into BOWs.

        - remove substitution tokens
        - remove punctuation
        - remove multiple whitespaces
        (- lemmatize   too time consuming for full dataset in given time)
        - remove stopwords
    """

    token_patt = r'(MAINCIT|CIT|FORMULA|REF|TABLE|FIGURE)'
    s = re.sub(token_patt, ' ', s)
    s = strip_punctuation(s)
    s = strip_multiple_whitespaces(s)
    s = remove_stopwords(s)
    return s.split()
