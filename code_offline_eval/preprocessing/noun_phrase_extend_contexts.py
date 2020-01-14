""" Extend a citation context CSV with noun phrases.
"""

import os
import re
import sys
from gensim import corpora

MAINCITS_PATT = re.compile(r'((CIT , )*MAINCIT( , CIT)*)')
CITS_PATT = re.compile(r'(((?<!MAIN)CIT , )*(?<!MAIN)CIT( , (?<!MAIN)CIT)*)')

ONLY_DIR_PRECEEDING = False
BOTH = False
PRECEEDING_MIN_LEN_2 = False


def merge_citation_token_lists(s):
    s = MAINCITS_PATT.sub('MAINCIT', s)
    s = CITS_PATT.sub('CIT', s)
    return s


def build(docs_path, dict_path):
    print('loading noun phrase dictionary')
    np_dictionary = corpora.Dictionary.load(dict_path)
    max_np_len = 0
    for np in np_dictionary.values():
        max_np_len = max(max_np_len, len(np.split()))

    total = sum(1 for line in open(docs_path))
    orig_n = os.path.splitext(docs_path)[0]
    ext_path = '{}_wNP.csv'.format(orig_n)
    with open(docs_path) as fi:
        with open(ext_path, 'w') as fo:
            for idx, line in enumerate(fi):
                if idx%10000 == 0:
                    print('{}/{} lines'.format(idx, total))
                vals = line.split('\u241E')
                if len(vals) == 4:
                    aid, adjacent, in_doc, text = vals
                    text = text.strip()
                    w_fos = False
                    w_pp = False
                elif len(vals) == 5:
                    aid, adjacent, in_doc, text, fos_annot = vals
                    fos_annot = fos_annot.strip()
                    w_fos = True
                    w_pp = False
                elif len(vals) == 6:
                    aid, adjacent, in_doc, text, fos_annot, pprep = vals
                    pprep = pprep.strip()
                    w_fos = True
                    w_pp = True
                else:
                    print('input file format can not be parsed\nexiting')
                    sys.exit()
                cntxt_nps = []
                cntxt_nps_prec = []
                # find NPs fast
                if ONLY_DIR_PRECEEDING:
                    text = merge_citation_token_lists(text)
                    try:
                        pre, post = text.split('MAINCIT')
                    except ValueError:
                        # data with no MAINCIT in context, assume sentence end
                        pre = text
                        post = ''
                    words = [re.sub('[^a-zA-Z0-9-]', ' ', w).strip()
                             for w in pre.split()]
                    # fixed position, so only go through sizes
                    for i in range(max_np_len):
                        chunk_size = max_np_len - i  # counting down
                        if chunk_size < 2 and PRECEEDING_MIN_LEN_2:
                            break
                        if len(words) < chunk_size:
                            continue
                        chunk = ' '.join(words[-chunk_size:])
                        try:
                            np_dictionary.token2id[chunk]
                            in_dict = True
                        except KeyError:
                            in_dict = False
                        if in_dict:
                            redundant = False
                            for already_in in cntxt_nps_prec:
                                if chunk in already_in:
                                    redundant = True
                                    break
                            if not redundant:
                                cntxt_nps_prec.append(chunk)
                if not ONLY_DIR_PRECEEDING or (ONLY_DIR_PRECEEDING and BOTH):
                    words = [re.sub('[^a-zA-Z0-9-]', ' ', w).strip()
                             for w in text.split()]
                    words = [w for w in words if len(w) > 0]
                    # for all lengths
                    for i in range(max_np_len):
                        chunk_size = max_np_len - i  # counting down
                        if len(words) < chunk_size:
                            continue
                        shift = 0
                        # comb through
                        while chunk_size + shift <= len(words):
                            chunk = ' '.join(words[shift:shift+chunk_size])
                            try:
                                np_dictionary.token2id[chunk]
                                in_dict = True
                            except KeyError:
                                in_dict = False
                            if in_dict:
                                redundant = False
                                for already_in in cntxt_nps:
                                    if chunk in already_in:
                                        redundant = True
                                        break
                                if not redundant:
                                    cntxt_nps.append(chunk)
                            shift += 1
                    # # find NPs slow
                    # for np in np_dictionary.values():
                    #     if ' {} '.format(np) in ' {} '.format(text):
                    #         cntxt_nps.append(np)
                if ONLY_DIR_PRECEEDING and not BOTH:
                    cntxt_nps = cntxt_nps_prec
                if not BOTH:
                    nprep = '\u241F'.join(cntxt_nps)
                if ONLY_DIR_PRECEEDING and BOTH:
                    rep_all = '\u241F'.join(cntxt_nps)
                    rep_prec = '\u241F'.join(cntxt_nps_prec)
                    # group seperator \u241D conceptually above record and unit
                    # seperator in hierarchy -- used here to avoid changing all
                    # existing code working just with record and unit seperator
                    nprep = '\u241D'.join([rep_all, rep_prec])
                if w_pp and w_fos:
                    new_vals = [aid, adjacent, in_doc, text, fos_annot, pprep, nprep]
                elif w_fos and not w_pp:
                    new_vals = [aid, adjacent, in_doc, text, fos_annot, nprep]
                else:
                    new_vals = [aid, adjacent, in_doc, text, nprep]
                ext_line = '{}\n'.format('\u241E'.join(new_vals))
                fo.write(ext_line)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(('usage: python3 *_extend_contexts.py </path/to/docs_file'
               '> </path/to/dictionary>'))
        sys.exit()
    docs_path = sys.argv[1]
    dict_path = sys.argv[2]
    build(docs_path, dict_path)
