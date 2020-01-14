""" Generate a Gensim dictionary for BoW, NP lists or claim tuples.
"""

import os
import json
import sys
from gensim import corpora
from util import bow_preprocess_string
from six import iteritems


def build(docs_path):
    """ Generate dict
    """

    texts = []
    pp_terms = []
    pp_terms_alt = []
    total = sum(1 for line in open(docs_path))
    with open(docs_path) as f:
        for idx, line in enumerate(f):
            if idx%10000 == 0:
                print('{}/{} lines'.format(idx, total))
            vals = line.split('\u241E')
            with_predpatt_rep = False
            with_predpatt_alt = False
            one_word_per_line_mode = False
            if len(vals) == 1:
                # to conveniently build gensim dicts from NP lists etc.
                text = vals[0].strip()
                one_word_per_line_mode = True
            elif len(vals) == 4:
                aid, adjacent, in_doc, text = vals
            elif len(vals) == 5:
                aid, adjacent, in_doc, text, fos_annot = vals
            elif len(vals) == 6:
                aid, adjacent, in_doc, text, fos_annot, pp_rep = vals
                with_predpatt_rep = True
            elif len(vals) == 7:
                aid, adjacent, in_doc, text, fos_annot, pp_rep, np_rep = vals
                with_predpatt_rep = True
            else:
                print('input file format can not be parsed\nexiting')
                sys.exit()
            if one_word_per_line_mode:
                texts.append([text])
            else:
                preprocessed_text = bow_preprocess_string(text)
                texts.append(preprocessed_text)
            if with_predpatt_rep:
                if '\u241F' in pp_rep:  # includes alternative version
                    with_predpatt_alt = True
                    pp_rep, pp_rep_alt = pp_rep.split('\u241F')
                    pp_lists_alt = json.loads(pp_rep_alt)
                    for weight, terms in pp_lists_alt:
                        pp_terms_alt.append(terms)
                pp_lists = json.loads(pp_rep)
                for weight, terms in pp_lists:
                    pp_terms.append(terms)
    docs_n = os.path.splitext(docs_path)[0]
    dict_tasks = [[texts, '{}.dict'.format(docs_n)]]
    if with_predpatt_rep:
        dict_tasks.append([pp_terms, '{}_PPterms.dict'.format(docs_n)])
        if with_predpatt_alt:
            dict_tasks.append([pp_terms_alt, '{}_PPterms_alt.dict'.format(docs_n)])
    for texts, save_fn in dict_tasks:
        print('building dictionary "{}"'.format(save_fn))
        dictionary = corpora.Dictionary(texts)
        if not one_word_per_line_mode:
            once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs)
                        if docfreq == 1]
            dictionary.filter_tokens(once_ids)
            dictionary.compactify()
        print('saving dictionary')
        dictionary.save(save_fn)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: python3 generate_gensim_dict.py </path/to/docs_file>')
        sys.exit()
    docs_path = sys.argv[1]
    build(docs_path)
