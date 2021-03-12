from extract_metrics_intra import ExtractMetricsIntra
from utils import Utils
from datetime import datetime
from visualization import Visualization
from cover_set import CoverSet


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
    # extractMetricsIntra.plt_histograms_intersect()

    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
    # intersect_df1, identity_group_dict_wos, sup_group_dict_wos, sub_group_dict_wos, intersect_group_dict_wos=extractMetricsIntra.find_groups(df1)
    # utils.save_obj(intersect_df1,'wos_num_intersections')

    # identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict=extractMetricsIntra.find_super_groups_and_intersection_all_journals_wos(df1)
    df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
    # intersect_df2, identity_group_dict_scopus, sup_group_dict_scopus, sub_group_dict_scopus, intersect_group_dict_scopus=extractMetricsIntra.find_groups(df2.T)
    # utils.save_obj(intersect_df2, 'scopus_num_intersections')
    # utils.save_obj(sub_group_dict_scopus, 'scopus_sub_group')
    # utils.save_obj(sup_group_dict_scopus, 'scopus_sup_group')
    # utils.save_obj(intersect_group_dict_scopus, 'scopus_intersect_group_dict')

    intersect_group_dict_scopus=utils.load_obj('scopus_intersect_group_dict')
    df_graph=extractMetricsIntra.prep_data_for_graph(intersect_group_dict_scopus)
    sub_group_dict_scopus=utils.load_obj('scopus_sub_group')
    # extractMetricsIntra.get_correlations_all_journals()
    # extractMetricsIntra.run_small_and_large_cats(df1,df2)
    # extractMetricsIntra.prep_data_for_venn_plots(df1, sub_group_dict_wos=None,intersect_group_dict_wos=None, scopus_df=df2, sub_group_dict_scopus=sub_group_dict_scopus, intersect_group_dict_scopus=intersect_group_dict_scopus)

    # extractMetricsIntra.create_journal_ranking_by_category(df1, df2.T)

    # extractMetricsIntra.get_categories_ranking_mismatch()



    # extractMetricsIntra.create_clusters_by_categories(df1, df2.T)
    # extractMetrics.calc_mean_sd()

    # extractMetricsIntra.analyse_categories_ranking_mismatch()

    # scopus_journals_dict= utils.load_obj('scopus_journals_dict')
    # df_scopus_cats_with_ranks=extractMetricsIntra.add_cats_data(df_scopus_cats_with_ranks, scopus_journals_dict)
    # utils.save_obj(df_scopus_cats_with_ranks, 'categories_with_ranks_df_scopus')

    # extractMetricsIntra.run_cover_set()
    df_scopus_cats_with_ranks = utils.load_obj("categories_with_ranks_df_scopus")
    scopus_categories_and_journals_dict=utils.load_obj("scopus_categories_and_journals_dict")

    # extractMetricsIntra.analyse_cover_set()

