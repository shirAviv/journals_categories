from extract_metrics_intra import ExtractMetricsIntra
from extract_metrics_inter import ExtractMetricsInter
from utils import Utils
from datetime import datetime
from visualization import Visualization
from cover_set import CoverSet
import pandas as pd

def run_intra_metrics():
    # extractMetricsIntra.plt_histograms_intersect()
    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
    df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
    df2=df2.T
    extractMetricsIntra.run_cover_set_all()
    # extractMetricsIntra.calc_mean_sd(df1,df2)

    # extractMetricsIntra.run_small_and_large_cats(df1, df2)

    # journals_dict_wos = utils.load_obj("wos_journals_dict_v3")
    # journals_dict_scopus = utils.load_obj("scopus_journals_dict_v2")
    # extractMetricsIntra.create_journals_with_cats_metrics('wos_journals_metrics.csv',journals_dict_wos,'scopus_scores_2019.csv',journals_dict_scopus)
    exit(0)
    # intersect_df1, identity_group_dict_wos, sup_group_dict_wos, sub_group_dict_wos, intersect_group_dict_wos = extractMetricsIntra.find_groups(
    #     df1)
    # utils.save_obj(intersect_group_dict_wos, 'wos_intersect_group_dict')

    # identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict=extractMetricsIntra.find_super_groups_and_intersection_all_journals_wos(df1)

    # intersect_df2, identity_group_dict_scopus, sup_group_dict_scopus, sub_group_dict_scopus, intersect_group_dict_scopus=extractMetricsIntra.find_groups(df2.T)
    # utils.save_obj(intersect_df2, 'scopus_num_intersections')
    # utils.save_obj(sub_group_dict_scopus, 'scopus_sub_group')
    # utils.save_obj(sup_group_dict_scopus, 'scopus_sup_group')
    # utils.save_obj(intersect_group_dict_scopus, 'scopus_intersect_group_dict')

    # intersect_group_dict = utils.load_obj('wos_intersect_group_dict')
    # graph_df, scopus_intersect_mat =extractMetricsIntra.prep_data_for_graph(intersect_group_dict_scopus)

    # utils.write_to_csv(graph_df[['category', 'intersecting category']], 'scopus_data_for_intersect_graph.csv')
    # utils.save_obj(scopus_intersect_mat,'scopus_intersects_for_graphs_above_0.1')
    # exit(0)
    # scopus_intersect_mat=utils.load_obj('scopus_intersects_for_graphs_above_0.1')
    # for cat, row in scopus_intersect_mat.iterrows():
    #     scopus_intersect_mat.loc[cat, cat] = 0
    # utils.write_to_csv(scopus_intersect_mat,'scopus_intersects_for_graphs_above_0.1.csv', index=True)

    # extractMetricsIntra.generate_graph(scopus_intersect_mat)
    # sub_group_dict_scopus=utils.load_obj('scopus_sub_group')
    # extractMetricsIntra.get_correlations_all_journals()

    extractMetricsIntra.prep_data_for_venn_plots(df1, sub_group_dict_wos=None,
                                                 intersect_group_dict_wos=intersect_group_dict, scopus_df=df2,
                                                 sub_group_dict_scopus=None, intersect_group_dict_scopus=None)

    # extractMetricsIntra.create_journal_ranking_by_category(df1, df2.T)

    # extractMetricsIntra.get_categories_ranking_mismatch()

    # extractMetricsIntra.create_clusters_by_categories(df1, df2.T)


    # extractMetricsIntra.analyse_categories_ranking_mismatch()

    # scopus_journals_dict= utils.load_obj('scopus_journals_dict')
    # df_scopus_cats_with_ranks=extractMetricsIntra.add_cats_data(df_scopus_cats_with_ranks, scopus_journals_dict)
    # utils.save_obj(df_scopus_cats_with_ranks, 'categories_with_ranks_df_scopus')

    # extractMetricsIntra.run_cover_set()
    df_scopus_cats_with_ranks = utils.load_obj("categories_with_ranks_df_scopus")
    scopus_categories_and_journals_dict = utils.load_obj("scopus_categories_and_journals_dict")

    # extractMetricsIntra.analyse_cover_set()

def run_inter_metrics():
    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
    df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
    df2 = df2.T


    #extractMetricsInter.analyse_gaps_in_cover_set()
    # extractMetricsInter.run_cover_set_all()
    # extractMetricsInter.run_correlations_inter()

    # identity_group_dict_wos_to_scopus, sup_group_dict_wos_to_scopus, sub_group_dict_wos_to_scopus, intersect_group_dict_wos_to_scopus=extractMetricsInter.run_groups_per_wos_cat(utils)

    # utils.save_obj(sub_group_dict_wos_to_scopus, 'wos_to_scopus_sub_group')
    # utils.save_obj(sup_group_dict_wos_to_scopus, 'wos_to_scopus_sup_group')
    # utils.save_obj(intersect_group_dict_wos_to_scopus, 'wos_to_scopus_intersect_group_dict')

    # identity_group_dict_scopus_to_wos, sup_group_dict_scopus_to_wos, sub_group_dict_scopus_to_wos, intersect_group_dict_scopus_to_wos = extractMetricsInter.run_groups_per_scopus_cat(utils)
    #
    # utils.save_obj(sub_group_dict_scopus_to_wos, 'scopus_to_wos_sub_group')
    # utils.save_obj(sup_group_dict_scopus_to_wos, 'scopus_to_wos_sup_group')
    # utils.save_obj(intersect_group_dict_scopus_to_wos, 'scopus_to_wos_intersect_group_dict')
    # sub_group_dict_wos_to_scopus=utils.load_obj('wos_to_scopus_sub_group')
    # sub_group_dict_wos_to_scopus=extractMetricsInter.remove_small_sub_groups_wos_to_scopus(sub_group_dict_wos_to_scopus)
    sub_group_dict_wos_to_scopus=utils.load_obj('wos_to_scopus_sub_group')
    sub_group_dict_scopus_to_wos=utils.load_obj('scopus_to_wos_sub_group')
    intersect_group_dict_wos_to_scopus=utils.load_obj('wos_to_scopus_intersect_group_dict')
    intersect_group_dict_scopus_to_wos = utils.load_obj('scopus_to_wos_intersect_group_dict')
    intersect_df = pd.DataFrame(columns=['categories', 'num intersecting categories'])
    for key, val in intersect_group_dict_wos_to_scopus.items():
        record = {'categories': key, 'num intersecting categories': int(len(val))}
        intersect_df = intersect_df.append(record, ignore_index=True)
    intersect_df['num intersecting categories'] = intersect_df['num intersecting categories'].apply(pd.to_numeric,
                                                                                                    errors='coerce')
    idx = intersect_df['num intersecting categories'].argmax()
    largest_intersecting_category = intersect_df.T[idx]['categories']
    num_largest_intersecting_category = intersect_df.T[idx]['num intersecting categories']
    print('wos to scopus largest intersecting category {}, num intersecting categories {}'.format(largest_intersecting_category,
                                                                                    num_largest_intersecting_category))

    intersect_df=intersect_df.iloc[0:0]
    for key, val in intersect_group_dict_scopus_to_wos.items():
        record = {'categories': key, 'num intersecting categories': int(len(val))}
        intersect_df = intersect_df.append(record, ignore_index=True)
    intersect_df['num intersecting categories'] = intersect_df['num intersecting categories'].apply(pd.to_numeric,
                                                                                                    errors='coerce')
    idx = intersect_df['num intersecting categories'].argmax()
    largest_intersecting_category = intersect_df.T[idx]['categories']
    num_largest_intersecting_category = intersect_df.T[idx]['num intersecting categories']
    print('scopus ro wos largest intersecting category {}, num intersecting categories {}'.format(
        largest_intersecting_category,
        num_largest_intersecting_category))

    # extractMetricsInter.prep_data_for_venn_plots(df1, sub_group_dict_wos=None,
    #                                              intersect_group_dict_wos=intersect_group_dict_wos_to_scopus, scopus_df=df2,
    #                                              sub_group_dict_scopus=sub_group_dict_scopus_to_wos, intersect_group_dict_scopus=intersect_group_dict_scopus_to_wos, extractMetrics=extractMetricsIntra)




if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils()
    vis=Visualization()
    cover_set=CoverSet(path=utils.path);
    extractMetricsIntra= ExtractMetricsIntra()
    extractMetricsIntra.utils=utils
    extractMetricsIntra.vis=vis
    extractMetricsIntra.cover_set=cover_set
    # run_intra_metrics()
    extractMetricsInter=ExtractMetricsInter()
    extractMetricsInter.utils = utils
    extractMetricsInter.vis = vis
    extractMetricsInter.cover_set = cover_set
    run_inter_metrics()


