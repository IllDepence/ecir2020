""" Extend a citation context CSV with PredPatt model representations.
"""

import os
import json
import sys
from predpatt_parse_contexts import build_sentence_representation

def build(docs_path):
    """ Extend CSV
    """

    total = sum(1 for line in open(docs_path))
    orig_n = os.path.splitext(docs_path)[0]
    ext_path = '{}_wPP_3.1.csv'.format(orig_n)
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
                elif len(vals) == 5:
                    aid, adjacent, in_doc, text, fos_annot = vals
                    fos_annot = fos_annot.strip()
                    w_fos = True
                else:
                    print('input file format can not be parsed\nexiting')
                    sys.exit()
                rep, rep_alt = build_sentence_representation(text)
                jrep = json.dumps(rep)  # currently v3.1
                jrep_alt = json.dumps(rep_alt)  # currently v3
                jreps = '\u241F'.join([jrep, jrep_alt])
                if w_fos:
                    new_vals = [aid, adjacent, in_doc, text, fos_annot, jreps]
                else:
                    new_vals = [aid, adjacent, in_doc, text, jreps]
                ext_line = '{}\n'.format('\u241E'.join(new_vals))
                fo.write(ext_line)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(('usage: python3 predpatt_extend_contexts.py </path/to/docs_file'
               '>'))
        sys.exit()
    docs_path = sys.argv[1]
    build(docs_path)
