""" Run citation re-prediction evaluation on a citation context CSV.

    (adjust train/test split in line 228 depending on data set)
"""

import json
import math
import operator
import sys
import numpy as np
from gensim import corpora, models, similarities
from util import bow_preprocess_string
from scipy.sparse import csc_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import LsiModel
from gensim.matutils import corpus2csc, Sparse2Corpus
from gensim.summarization.bm25 import BM25

SILENT = False
AT_K = 10


def prind(s):
    if not SILENT:
        print(s)


def better_rank(sims_a, sims_b, train_mids, test_mid):
    sims_list_a = list(enumerate(sims_a))
    sims_list_a.sort(key=lambda tup: tup[1], reverse=True)
    ranking_a = [s[0] for s in sims_list_a]
    sims_list_b = list(enumerate(sims_b))
    sims_list_b.sort(key=lambda tup: tup[1], reverse=True)
    ranking_b = [s[0] for s in sims_list_b]

    if sims_list_b[0][1] == 0:
        return False, -1, -1

    rank_a = len(ranking_a)
    for idx, doc_id in enumerate(ranking_a):
        if train_mids[doc_id] == test_mid:
            rank_a = idx+1
            break
    rank_b = len(ranking_b)
    for idx, doc_id in enumerate(ranking_b):
        if train_mids[doc_id] == test_mid:
            rank_b = idx+1
            break
    if rank_a > rank_b:
        return 1, rank_a, rank_b
    else:
        return 0, rank_a, rank_b


def fos_boost_ranking(bow_ranking, fos_boost, top_dot_prod):
    if top_dot_prod < 1:
        return bow_ranking
    point_dic = {}
    for i in range(len(bow_ranking)):
        points = len(bow_ranking) - i
        if bow_ranking[i] not in point_dic:
            point_dic[bow_ranking[i]] = 0
        point_dic[bow_ranking[i]] += points
    for boost in fos_boost:
        point_dic[boost] += 1.1
    comb = sorted(point_dic.items(), key=operator.itemgetter(1), reverse=True)
    return [c[0] for c in comb]


def combine_simlists(sl1, sl2, weights):
    sl = []
    for i in range(len(sl1)):
        sl.append(
            (sl1[i]*weights[0])+
            (sl2[i]*weights[1])
            )
    return sl


def sum_weighted_term_lists(wtlist, dictionary):
    if len(wtlist) == 0:
        return []
    term_vecs = []
    for weight, terms in wtlist:
        term_vec_raw = dictionary.doc2bow(terms)
        term_vec = [(term_id, weight*val) for term_id, val in term_vec_raw]
        term_vecs.append(term_vec)
    # make into numpy matrix for convenience
    term_matrix = corpus2csc(term_vecs)
    # calculate sum
    sum_vec = Sparse2Corpus(
        csc_matrix(term_matrix.sum(1))
        )[0]
    return sum_vec


def recommend(docs_path, dict_path, use_fos_annot=False, pp_dict_path=None,
              np_dict_path=None, lda_preselect=False,
              combine_train_contexts=True):
    """ Recommend
    """

    test = []
    train_mids = []
    train_texts = []
    train_foss = []
    train_ppann = []
    train_nps = []
    foss = []
    tmp_bag = []
    adjacent_cit_map = {}

    if pp_dict_path and False:
        prind('loading predpatt dictionary')
        pp_dictionary = corpora.Dictionary.load(pp_dict_path)
        pp_num_unique_tokens = len(pp_dictionary.keys())
        use_predpatt_model = True
        if not combine_train_contexts:
            prind(('usage of predpatt model is not implemented for not'
                   'combining train contexts.\nexiting.'))
            sys.exit()
    else:
        use_predpatt_model = False
        pp_dictionary = None

    if np_dict_path:
        prind('loading noun phrase dictionary')
        np_dictionary = corpora.Dictionary.load(np_dict_path)
        np_num_unique_tokens = len(np_dictionary.keys())
        use_noun_phrase_model = True
    else:
        use_noun_phrase_model = False
        np_dictionary = None

    prind('checking file length')
    num_lines = sum(1 for line in open(docs_path))

    # # for MAG eval
    # mag_id2year = {}
    # with open('MAG_CS_en_year_map.csv') as f:
    #     for line in f:
    #         pid, year = line.strip().split(',')
    #         mag_id2year[pid] = int(year)
    # # /for MAG eval

    prind('train/test splitting')
    with open(docs_path) as f:
        for idx, line in enumerate(f):
            if idx == 0:
                tmp_bag_current_mid = line.split('\u241E')[0]
            if idx%10000 == 0:
                prind('{}/{} lines'.format(idx, num_lines))
            cntxt_foss = []
            cntxt_ppann = []
            cntxt_nps = []
            # handle varying CSV formats
            vals = line.split('\u241E')
            if use_noun_phrase_model:
                cntxt_nps = vals[-1]
                if '\u241D' in cntxt_nps:  # includes NP<marker> variant
                    np_all, np_marker = cntxt_nps.split('\u241D')
                    cntxt_nps = np_marker  # mby use both for final eval
                cntxt_nps = [np for np in cntxt_nps.strip().split('\u241F')]
                vals = vals[:-1]
            if len(vals) == 4:
                mid, adjacent, in_doc, text = vals
            elif len(vals) == 5:
                if use_predpatt_model:
                    mid, adjacent, in_doc, text, pp_annot_json = vals
                else:
                    mid, adjacent, in_doc, text, fos_annot = vals
            elif len(vals) == 6:
                mid, adjacent, in_doc, text, fos_annot, pp_annot_json = vals
            else:
                prind('input file format can not be parsed\nexiting')
                sys.exit()
            if len(vals) in [5, 6] and use_fos_annot:
                cntxt_foss = [f.strip() for f in fos_annot.split('\u241F')
                              if len(f.strip()) > 0]
                foss.extend(cntxt_foss)
            if use_predpatt_model:
                if '\u241F' in pp_annot_json:  # includes alternative version
                    ppann, ppann_alt = pp_annot_json.split('\u241F')
                    pp_annot_json = ppann
                cntxt_ppann = json.loads(pp_annot_json)
            # create adjacent map for later use in eval
            if mid not in adjacent_cit_map:
                adjacent_cit_map[mid] = []
            if len(adjacent) > 0:
                adj_cits = adjacent.split('\u241F')
                for adj_cit in adj_cits:
                    if adj_cit not in adjacent_cit_map[mid]:
                        adjacent_cit_map[mid].append(adj_cit)
            # fill texts
            if mid != tmp_bag_current_mid or idx == num_lines-1:
                # tmp_bag now contains all lines sharing ID tmp_bag_current_mid
                num_contexts = len(tmp_bag)
                sub_bags_dict = {}
                for item in tmp_bag:
                    item_in_doc = item[0]
                    item_text = item[1]
                    item_foss = item[2]
                    item_ppann = item[3]
                    item_nps = item[4]
                    if item_in_doc not in sub_bags_dict:
                        sub_bags_dict[item_in_doc] = []
                    sub_bags_dict[item_in_doc].append(
                        [item_text, item_foss, item_ppann, item_nps]
                        )
                if len(sub_bags_dict) < 2:
                    # can't split, reset bag, next
                    tmp_bag = []
                    tmp_bag_current_mid = mid
                    continue
                order = sorted(sub_bags_dict,
                               key=lambda k: len(sub_bags_dict[k]),
                               reverse=True)
                # ↑ keys for sub_bags_dict, ordered for largest bag to smallest

                min_num_train = math.floor(num_contexts * 0.8)
                train_tups = []
                test_tups = []
                for jdx, sub_bag_key in enumerate(order):
                    sb_tup = sub_bags_dict[sub_bag_key]
                    # if sub_bag_key[1:3] == '06':  # time split ACL
                    # if mag_id2year[sub_bag_key] > 2017:  # time split MAG
                    # if sub_bag_key[:2] == '17':  # time split arXiv
                    if len(train_tups) > min_num_train or jdx == len(order)-1:
                        test_tups.extend(sb_tup)
                    else:
                        train_tups.extend(sb_tup)
                test.extend(
                    [
                        [tmp_bag_current_mid,                            # mid
                         tup[0],                                         # text
                         tup[1],                                         # fos
                         sum_weighted_term_lists(tup[2], pp_dictionary), # pp
                         tup[3]                                          # nps
                        ]
                    for tup in test_tups
                    ])
                if combine_train_contexts:
                    # combine train contexts per cited doc
                    train_text_combined = ' '.join(tup[0] for tup in train_tups)
                    train_mids.append(tmp_bag_current_mid)
                    train_texts.append(train_text_combined.split())
                    train_foss.append(
                        [fos for tup in train_tups for fos in tup[1]]
                        )
                    train_ppann.append(
                        sum_weighted_term_lists(
                            sum([tup[2] for tup in train_tups], []),
                            pp_dictionary
                            )
                        )
                    train_nps.append(
                        [np for tup in train_tups for np in tup[3]]
                        )
                else:
                    # don't combine train contexts per cited doc
                    for tup in train_tups:
                        train_mids.append(tmp_bag_current_mid)
                        train_texts.append(tup[0].split())
                        train_foss.append([fos for fos in tup[1]])
                        train_nps.append([np for np in tup[1]])
                # reset bag
                tmp_bag = []
                tmp_bag_current_mid = mid
            tmp_bag.append([in_doc, text, cntxt_foss, cntxt_ppann, cntxt_nps])
    prind('loading dictionary')
    dictionary = corpora.Dictionary.load(dict_path)
    num_unique_tokens = len(dictionary.keys())
    prind('building corpus')
    corpus = [dictionary.doc2bow(text) for text in train_texts]

    if use_fos_annot:
        prind('preparing FoS model')
        mlb = MultiLabelBinarizer()
        mlb.fit([foss])
        train_foss_matrix = mlb.transform(train_foss)
        train_foss_set_sizes = np.sum(train_foss_matrix, 1)
    prind('generating TFIDF model')
    tfidf = models.TfidfModel(corpus)
    prind('preparing similarities')
    index = similarities.SparseMatrixSimilarity(
                tfidf[corpus],
                num_features=num_unique_tokens)

    bm25 = BM25(corpus)
    average_idf = sum(
        map(lambda k: float(bm25.idf[k]),
            bm25.idf.keys())
        ) / len(bm25.idf.keys())

    if lda_preselect:
        orig_index = index.index.copy()

        prind('generating LDA/LSI model')
        lda = LsiModel(tfidf[corpus], id2word=dictionary, num_topics=100)
        prind('preparing similarities')
        lda_index = similarities.SparseMatrixSimilarity(
                    lda[tfidf[corpus]],
                    num_features=num_unique_tokens)

    if use_predpatt_model:
        prind('preparing claim similarities')
        pp_tfidf = models.TfidfModel(train_ppann)
        pp_index = similarities.SparseMatrixSimilarity(
            pp_tfidf[train_ppann],
            num_features=pp_num_unique_tokens)

    if use_noun_phrase_model:
        prind('preparing noun phrase similarities')
        np_corpus = [np_dictionary.doc2bow(nps) for nps in train_nps]
        np_index = similarities.SparseMatrixSimilarity(
            np_corpus,
            num_features=np_num_unique_tokens)

    # models: BoW, NP<marker>, Claim, Claim+BoW
    eval_models = [
        {'name':'bow'},
        {'name':'np'},
        {'name':'claim'},
        {'name':'claim+bow'}
        ]
    for mi in range(len(eval_models)):
        eval_models[mi]['num_cur'] = 0
        eval_models[mi]['num_top'] = 0
        eval_models[mi]['num_top_5'] = 0
        eval_models[mi]['num_top_10'] = 0
        eval_models[mi]['ndcg_sums'] = [0]*AT_K
        eval_models[mi]['map_sums'] = [0]*AT_K
        eval_models[mi]['mrr_sums'] = [0]*AT_K
        eval_models[mi]['recall_sums'] = [0]*AT_K
    prind('test set size: {}\n- - - - - - - -'.format(len(test)))
    for test_item_idx, tpl in enumerate(test):
        if test_item_idx > 0 and test_item_idx%10000 == 0:
            save_results(
                docs_path, num_lines, len(test), eval_models, suffix='_tmp'
                )
        test_mid = tpl[0]
        # if test_mid not in train_mids:
        #     # not testable
        #     continue
        test_text = bow_preprocess_string(tpl[1])
        if use_fos_annot:
            test_foss_vec = mlb.transform([tpl[2]])
            dot_prods = train_foss_matrix.dot(
                test_foss_vec.transpose()
                ).transpose()[0]
            with np.errstate(divide='ignore',invalid='ignore'):
                fos_sims = np.nan_to_num(dot_prods/train_foss_set_sizes)
            fos_sims_list = list(enumerate(fos_sims))
            fos_sims_list.sort(key=lambda tup: tup[1], reverse=True)
            fos_ranking = [s[0] for s in fos_sims_list]
            fos_boost = np.where(
                dot_prods >= dot_prods.max()-1
                )[0].tolist()
            top_dot_prod = dot_prods[-1]
        if use_predpatt_model:
            pp_sims = pp_index[pp_tfidf[tpl[3]]]
            pp_sims_list = list(enumerate(pp_sims))
            pp_sims_list.sort(key=lambda tup: tup[1], reverse=True)
            pp_ranking = [s[0] for s in pp_sims_list]
        if use_noun_phrase_model:
            np_sims = np_index[np_dictionary.doc2bow(tpl[4])]
            np_sims_list = list(enumerate(np_sims))
            np_sims_list.sort(key=lambda tup: tup[1], reverse=True)
            np_ranking = [s[0] for s in np_sims_list]
        test_bow = dictionary.doc2bow(test_text)
        if lda_preselect:
            # pre select in LDA/LSI space
            lda_sims = lda_index[lda[tfidf[test_bow]]]
            lda_sims_list = list(enumerate(lda_sims))
            lda_sims_list.sort(key=lambda tup: tup[1], reverse=True)
            lda_ranking = [s[0] for s in lda_sims_list]
            lda_picks = lda_ranking[:1000]
            index.index = orig_index[lda_picks]
        sims = index[tfidf[test_bow]]
        sims_list = list(enumerate(sims))
        sims_list.sort(key=lambda tup: tup[1], reverse=True)
        bow_ranking = [s[0] for s in sims_list]

        bm25_scores = list(enumerate(bm25.get_scores(test_bow, average_idf)))
        bm25_scores.sort(key=lambda tup: tup[1], reverse=True)
        bm25_ranking = [s[0] for s in bm25_scores]

        if lda_preselect:
            # translate back from listing in LDA/LSI pick subset to global listing
            bow_ranking = [lda_picks[r] for r in bow_ranking]
        if use_fos_annot:
            boost_ranking = fos_boost_ranking(
                bow_ranking, fos_boost, top_dot_prod)
        if not combine_train_contexts:
            seen = set()
            seen_add = seen.add
            final_ranking = [x for x in final_ranking
                     if not (train_mids[x] in seen or seen_add(train_mids[x]))]
        if use_predpatt_model:
            sims_comb = combine_simlists(sims, pp_sims, [2, 1])
            comb_sims_list = list(enumerate(sims_comb))
            comb_sims_list.sort(key=lambda tup: tup[1], reverse=True)
            comb_ranking = [s[0] for s in comb_sims_list]

        for mi in range(len(eval_models)):
            if mi == 0:
                final_ranking = bow_ranking
            elif mi == 1:
                final_ranking = np_ranking
            elif mi == 2:
                final_ranking = pp_ranking
            elif mi == 3:
                final_ranking = comb_ranking
            rank = len(bow_ranking)  # assign worst possible
            for idx, doc_id in enumerate(final_ranking):
                if train_mids[doc_id] == test_mid:
                    rank = idx+1
                    break
                if idx >= 10:
                   break
            dcgs = [0]*AT_K
            idcgs = [0]*AT_K
            precs = [0]*AT_K
            num_rel_at = [0]*AT_K
            num_rel = 1 + len(adjacent_cit_map[test_mid])
            num_rel_at_k = 0
            for i in range(AT_K):
                relevant = False
                placement = i+1
                doc_id = final_ranking[i]
                result_mid = train_mids[doc_id]
                if result_mid == test_mid:
                    relevance = 1
                    num_rel_at_k += 1
                    relevant = True
                elif result_mid in adjacent_cit_map[test_mid]:
                    relevance = .5
                    num_rel_at_k += 1
                    relevant = True
                else:
                    relevance = 0
                num_rel_at[i] = num_rel_at_k
                if relevant:
                    precs[i] = num_rel_at_k / placement
                denom = math.log2(placement + 1)
                dcg_numer = math.pow(2, relevance) - 1
                for j in range(i, AT_K):
                    dcgs[j] += dcg_numer / denom
                if placement == 1:
                    ideal_rel = 1
                elif placement <= num_rel:
                    ideal_rel = .5
                else:
                    ideal_rel = 0
                idcg_numer = math.pow(2, ideal_rel) - 1
                for j in range(i, AT_K):
                    # note this^    we go 0~9, 1~9, 2~9, ..., 9
                    idcgs[j] += idcg_numer / denom
            for i in range(AT_K):
                eval_models[mi]['ndcg_sums'][i] += dcgs[i] / idcgs[i]
                eval_models[mi]['map_sums'][i] += sum(precs[:i+1])/max(num_rel_at[i], 1)
                if rank <= i+1:
                    eval_models[mi]['mrr_sums'][i] += 1 / rank
                    eval_models[mi]['recall_sums'][i] += 1
            if rank == 1:
                eval_models[mi]['num_top'] += 1
            if rank <= 5:
                eval_models[mi]['num_top_5'] += 1
            if rank <= 10:
                eval_models[mi]['num_top_10'] += 1
            eval_models[mi]['num_cur'] += 1
            prind('- - - - - {}/{} - - - - -'.format(
                eval_models[0]['num_cur'], len(test))
                )
            prind('#1: {}'.format(eval_models[0]['num_top']))
            prind('in top 5: {}'.format(eval_models[0]['num_top_5']))
            prind('in top 10: {}'.format(eval_models[0]['num_top_10']))
            prind('ndcg@5: {}'.format(
                eval_models[0]['ndcg_sums'][4]/eval_models[0]['num_cur'])
                )
            prind('map@5: {}'.format(
                eval_models[0]['map_sums'][4]/eval_models[0]['num_cur'])
                )
            prind('mrr@5: {}'.format(
                eval_models[0]['mrr_sums'][4]/eval_models[0]['num_cur'])
                )
            prind('recall@5: {}'.format(
                eval_models[0]['recall_sums'][4]/eval_models[0]['num_cur'])
                )

    for mi in range(len(eval_models)):
        eval_models[mi]['num_applicable'] = eval_models[mi]['num_cur']
        eval_models[mi]['ndcg_results'] = [
            sm/eval_models[mi]['num_cur'] for sm in eval_models[mi]['ndcg_sums']
            ]
        eval_models[mi]['map_results'] = [
            sm/eval_models[mi]['num_cur'] for sm in eval_models[mi]['map_sums']
            ]
        eval_models[mi]['mrr_results'] = [
            sm/eval_models[mi]['num_cur'] for sm in eval_models[mi]['mrr_sums']
            ]
        eval_models[mi]['recall_results'] = [
            sm/eval_models[mi]['num_cur'] for sm in eval_models[mi]['recall_sums']
            ]

    return eval_models, num_lines, len(test)


def save_results(docs_path, num_lines, num_test, eval_models, suffix=''):
    timestamp = int(time.time())
    result_file_name = 'eval_results_{}{}.json'.format(timestamp, suffix)
    result_data = {
        'data': docs_path,
        'num_contexts': num_lines,
        'num_test_set_items': num_test,
        'models': eval_models,
        }
    with open(result_file_name, 'w') as f:
        f.write(json.dumps(result_data))


if __name__ == '__main__':
    if len(sys.argv) not in [3, 4, 5]:
        prind(('usage: python3 evaluate.py </path/to/context_csv> </path/to/ge'
               'nsim_dict> [<claim_gensim_dict>] [<np_gensim_dict>]'))
        sys.exit()
    docs_path = sys.argv[1]
    dict_path = sys.argv[2]
    pp_dict_path = None
    np_dict_path = None
    if len(sys.argv) >= 4:
        pp_dict_path = sys.argv[3]
    if len(sys.argv) == 5:
        np_dict_path = sys.argv[4]

    eval_models, num_lines, num_test = recommend(
        docs_path,
        dict_path,
        pp_dict_path=pp_dict_path,
        np_dict_path=np_dict_path,
        )

    save_results(docs_path, num_lines, num_test, eval_models)
