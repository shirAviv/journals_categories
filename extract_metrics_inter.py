from scipy import stats
import pandas as pd
from extract_metrics_intra import ExtractMetricsIntra


class ExtractMetricsInter:

    def run_cover_set_per_cat(self):
        df1 = self.utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')

        df2 = self.utils.load_obj('scopus_to_wos_categories_for_group_mapping')

        # for wos_category, row in df1.iterrows():
        #     to_remove=row['journals'].loc[row['journals']['Scopus Journal title']=='']
        #     if len(to_remove)>0:
        #         print(to_remove)
        #         index_to_remove=to_remove.index
        #         row['journals'].drop(index=index_to_remove, inplace=True)
        #     print(wos_category)
        # utils.save_obj(df1,'wos_to_scopus_categories_for_group_mapping_v1')
        # df_wos_cover_set = self.cover_set.run_cover_set_per_category_wos(df1)
        # self.utils.save_obj(df_wos_cover_set,'cover_set_wos_to_scopus')
        df_wos_cover_set = self.utils.load_obj('cover_set_wos_to_scopus')
        self.vis.plot_cover_set_inter(df_wos_cover_set, "WOS categories cover set by Scopus - cumulative")

        # for scopus_category, row in df2.T.iterrows():
        #     to_remove=row['journals'].loc[row['journals']['Scopus Journal title']=='']
        # df_scopus_cover_set=self.cover_set.run_cover_set_per_category_scopus(df2.T)
        # self.utils.save_obj(df_scopus_cover_set,'cover_set_scopus_to_wos')
        df_scopus_cover_set=self.utils.load_obj('cover_set_scopus_to_wos')
        self.vis.plot_cover_set_inter(df_scopus_cover_set,"Scopus categories cover set by WOS - cumulative")
        # df_scopus_cover_set_no_outlier=df_scopus_cover_set[df_scopus_cover_set["Num journals"] < 1000]

        self.vis.plt_coverset_size(df_wos_cover_set, df_scopus_cover_set_no_outlier,
                                   label1='Scopus categories \nmin cover of \nWOS categories',
                                   label2='WOS categories \nmin cover of \nScopus categories',
                                   title1="Minimal cover size \n by number of journals",
                                   title2= "Minimal cover size \n by number of covering categories",cover_set_method='ILP')
        self.vis.plt_coverset_size(df_wos_cover_set, df_scopus_cover_set_no_outlier,
                                   label1='Scopus categories \nmin cover of \nWOS categories',
                                   label2='WOS categories \nmin cover of \nScopus categories',
                                   title1="Minimal cover size \n by number of journals",
                                   title2='Minimal cover size \n by number of covering categories', cover_set_method='ILP', extract_low=20)

        self.vis.plt_coverset_size(df_wos_cover_set, df_scopus_cover_set_no_outlier,
                                   label1='Scopus categories \nmin cover of \nWOS categories',
                                   label2='WOS categories \nmin cover of \nScopus categories',
                                   title1="Minimal cover size \n by number of journals",
                                   title2='Minimal cover size \n by number of covering categories',
                                   cover_set_method='ILP', extract_high=20)

    def run_cover_set_all(self):
        # df1 = self.utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')

        # df2 = self.utils.load_obj('scopus_to_wos_categories_for_group_mapping')

        sco_cats_dict = self.utils.load_obj('no_cat_wos_to_scopus_categories_for_group_mapping')
        record_wos=self.cover_set.run_cover_set_no_cat_wos(sco_cats_dict=sco_cats_dict)
        print(record_wos)

        wos_cats_dict = self.utils.load_obj('no_cat_scopus_to_wos_categories_for_group_mapping')
        del wos_cats_dict['']
        record_scopus=self.cover_set.run_cover_set_no_cat_scopus(cats_dict=wos_cats_dict)
        print(record_scopus)

    def analyse_gaps_in_cover_set(self):
        df1 = self.utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')

        df2 = self.utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_scopus = self.utils.load_obj("scopus_journals_with_metrics")

        df_wos_cover_set = self.utils.load_obj('cover_set_wos_to_scopus')
        df_scopus_cover_set = self.utils.load_obj('cover_set_scopus_to_wos')
        df_wos_cover_set.sort_values(by='Min Cover set ILP', inplace=True)
        min_cover_set_list=df_wos_cover_set['Min Cover set ILP']
        print("Min cover set WOS to Scopus range {} - {}".format(min_cover_set_list.min(), min_cover_set_list.max()))
        df_scopus_cover_set.sort_values(by='Min Cover set ILP', inplace=True)
        min_cover_set_list = df_scopus_cover_set['Min Cover set ILP']
        print("Min cover set Scopus to WOS range {} - {}".format(min_cover_set_list.min(), min_cover_set_list.max()))

        df_wos_cover_set['cover_set_gap'] = df_wos_cover_set['Min Cover set ILP'] - df_wos_cover_set[
            'Min cover set Greedy 90']
        df_wos_cover_set.sort_values(by='cover_set_gap',ascending=False, inplace=True)
        largest_gap_wos_categories=df_wos_cover_set.loc[df_wos_cover_set['cover_set_gap'] > 8, 'Category'].values
        journals_not_covered_df=pd.DataFrame()
        for wos_category in largest_gap_wos_categories:
        # for wos_category, row in df.iterrows():
            sco_cats_dict=df1['scopus_categories'][wos_category]
            if isinstance(sco_cats_dict, float):
                print('skipping cat {}'.format(wos_category))
                continue
            journals = set(df1['journals'][wos_category]['Scopus Journal title'])
            if '' in journals:
                journals.remove('')
            cover_set_greedy_90, journals_not_covered = self.cover_set.cover_set_greedy(journals_set=journals, cats_dict=sco_cats_dict, threshold=0.9, ret_journals_not_covered=True)
            journals_not_covered_df=journals_not_covered_df.append(journals_with_cats_metrics_scopus.loc[
            journals_with_cats_metrics_scopus['Journal name'].isin(journals_not_covered)])
        print(len(journals_not_covered_df))
        categories_with_ranks_df_scopus = self.utils.load_obj('categories_with_ranks_df_scopus')




    def run_correlations_inter(self):
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_scopus = self.utils.load_obj("scopus_journals_with_metrics")
        self.get_inter_system_correlations(journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus)

    def get_inter_system_correlations(self,journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus):
        print('Calculating pearson correlation for Scopus and wos num categories')
        # df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['SJR']) == False]
        r, pValue = stats.pearsonr(journals_with_cats_metrics_wos['num categories'].values, journals_with_cats_metrics_scopus['num categories'].values)
        print('Pearsons correlation categories wos and scopus: r {}. pValue {}'.format(r, pValue))

        print('Calculating wilcoxson rank test correlation for Scopus and wos num categories')
        stat, pValue = stats.wilcoxon(journals_with_cats_metrics_wos['num categories'].values, journals_with_cats_metrics_scopus['num categories'].values)
        print('Wilcoxson rank test - categories wos and scopus: stat {}. pValue {}'.format(stat, pValue))

        print('Calculating paired t-test for Scopus and wos num categories')
        stat, pValue = stats.ttest_rel(journals_with_cats_metrics_wos['num categories'].values,
                                      journals_with_cats_metrics_scopus['num categories'].values)
        print('t-test - categories wos and scopus: stat {}. pValue {}'.format(stat, pValue))

    def run_groups_per_wos_cat(self, utils):
        df_wos = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
        df_scopus = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict = self.find_super_groups_and_intersection_per_cat(df_wos, df_scopus.T)
        return identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict

    def run_groups_per_scopus_cat(self, utils):
        df_wos = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
        df_scopus = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict = self.find_super_groups_and_intersection_per_cat(df_scopus.T, df_wos, from_scopus=True)
        return identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict



    def find_super_groups_and_intersection_per_cat(self, df_cats_from, df_cats_to, from_scopus=False):

        sup_group_dict = dict()
        intersect_group_dict = dict()
        identity_group_dict = dict()
        sub_group_dict = dict()
        for from_category, row in df_cats_from.iterrows():
            if from_scopus:
                to_cats_dict = row['wos_categories']
                journals = set(row['journals'])
            else:
                to_cats_dict = row['scopus_categories']
                journals = set(row['journals']['Scopus Journal title'].unique())
            if isinstance(to_cats_dict, float):
                continue


            for to_cat in to_cats_dict.keys():
                if from_scopus:
                    if len(to_cat)==0:
                        continue
                    compared_journals=set(df_cats_to['journals'][to_cat]['Scopus Journal title'].unique())
                else:
                    compared_journals = set(df_cats_to['journals'][to_cat])
                if journals.isdisjoint(compared_journals):
                    continue
                if journals.issubset(compared_journals) and journals.issuperset(compared_journals):
                    if not from_category in identity_group_dict.keys():
                        identity_group_dict[from_category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    journals_list = journals
                    ratio = len(journals_list) / len(compared_journals)
                    record = {'category': to_cat, 'journals': journals_list, 'ratio': ratio}
                    identity_group_dict[from_category] = identity_group_dict[from_category].append(record, ignore_index=True)
                    continue
                if journals.issubset(compared_journals):
                    if not from_category in sup_group_dict.keys():
                        sup_group_dict[from_category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    journals_list = journals
                    ratio = len(journals_list) / len(compared_journals)
                    record = {'category': to_cat, 'journals': journals_list, 'ratio': ratio}
                    sup_group_dict[from_category] = sup_group_dict[from_category].append(record, ignore_index=True)
                    continue
                if journals.issuperset(compared_journals):
                    if not from_category in sub_group_dict.keys():
                        sub_group_dict[from_category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    journals_list = compared_journals
                    ratio = len(journals_list) / len(journals)
                    record = {'category': to_cat, 'journals': journals_list, 'ratio': ratio}
                    sub_group_dict[from_category] = sub_group_dict[from_category].append(record, ignore_index=True)
                    continue
                if not from_category in intersect_group_dict.keys():
                    intersect_group_dict[from_category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                journals_list = journals.intersection(compared_journals)
                ratio = len(journals_list) / len(journals)
                record = {'category': to_cat, 'journals': journals_list, 'ratio': ratio}
                intersect_group_dict[from_category] = intersect_group_dict[from_category].append(record, ignore_index=True)

        return identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict

    # def find_sup_group(self, groups_dict):
    #     same_groups_list=[]
    #     for k,v in groups_dict.items():
    #         groups=v['groups']['sup_group']
    #         if len(groups)>0:
    #             sup_groups=groups
    #             print('cat {} has sup cats {}'.format(k,sup_groups))
    #             same_groups_list.append(sup_groups)


    # def find_super_groups_and_intersection_all_journals(self, sco_cats_dict):
    #     sorted_sco_cats = sorted(sco_cats_dict, key=lambda k: len(sco_cats_dict[k]))
    #     for i in range(len(sorted_sco_cats)):
    #         sco_cat = sorted_sco_cats[i]
    #         sco_cat_dict = dict()
    #         sco_cat_dict['sup_group'] = []
    #         sco_cat_dict['intersect_group'] = []
    #         sco_cat_dict['same_group'] = []
    #         journals = sco_cats_dict[sorted_sco_cats[i]]
    #
    #         for j in range(i + 1, len(sorted_sco_cats)):
    #             intersect = False
    #             sup_group = True
    #             compared_journals = sco_cats_dict[sorted_sco_cats[j]]
    #             for journal in journals:
    #                 if journal in compared_journals:
    #                     intersect = True
    #                 else:
    #                     sup_group = False
    #             if intersect == True and sup_group == True:
    #                 if len(journals) == len(compared_journals):
    #                     sco_cat_dict['same_group'].append(sorted_sco_cats[j])
    #                 else:
    #                     sco_cat_dict['sup_group'].append(sorted_sco_cats[j])
    #             else:
    #                 if intersect == True:
    #                     sco_cat_dict['intersect_group'].append(sorted_sco_cats[j])
    #         sco_cats_dict[sco_cat] = dict()
    #         sco_cats_dict[sco_cat]['journals'] = journals
    #         sco_cats_dict[sco_cat]['groups'] = sco_cat_dict
    #     return sco_cats_dict

    def prep_data_for_venn_plots(self,wos_df,sub_group_dict_wos, intersect_group_dict_wos, scopus_df, sub_group_dict_scopus,intersect_group_dict_scopus, extractMetrics ):
        # extractMetrics.prep_data_for_venn_subset(sub_group_dict_scopus)
        extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.90,wos_df.T, scopus_df.T)
        # print('0.8')
        # extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.8, wos_df.T)

        # self.prep_data_for_venn_subset(sub_group_dict_scopus)
        extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_scopus, 0.85, scopus_df.T, wos_df.T)

    def remove_small_sub_groups_wos_to_scopus(self, wos_to_scopus_sub_groups_dict):
        small_cats=list(["Emergency Medical Services","Dental Hygiene","Podiatry","Nurse Assisting","Drug Guides",
                         "Immunology and Microbiology (miscellaneous)","Review and Exam Preparation","Pharmacology (nursing)"])
        cats_to_remove=[]
        for cat, sub_cats_df in wos_to_scopus_sub_groups_dict.items():
            idxes_to_remove=[]
            for idx,row in sub_cats_df.iterrows():
                if row['category'] in small_cats:
                    idxes_to_remove.append(idx)
            sub_cats_df.drop(index=idxes_to_remove, inplace=True)
            if len(sub_cats_df)==0:
                cats_to_remove.append(cat)
        for cat in cats_to_remove:
            del wos_to_scopus_sub_groups_dict[cat]
        return wos_to_scopus_sub_groups_dict


    def run_thresholds(self, utils, psj=None, pwj=None, vis=None):
        mapping_scop_to_wos = utils.load_obj('scopus_to_wos_categories_mapping')
        df_thresholds_scop_to_wos = self.extract_matches(mapping_df=mapping_scop_to_wos)
        mapping_wos_to_scop = utils.load_obj('wos_to_scopus_categories_mapping')
        df_thresholds_wos_to_scop = self.extract_matches(mapping_df=mapping_wos_to_scop)
        self.vis.plt_match_by_threshold(df_thresholds_wos_to_scop, df_thresholds_scop_to_wos,
                                   'Categories match by threshold')

    def extract_matches(self, mapping_df):
        totals_df=mapping_df.loc['Total'].copy()
        mapping_df.drop(index='Total', inplace=True)
        maxValuesObj = mapping_df.max()
        maxValueIndexObj = mapping_df.idxmax()
        a=maxValueIndexObj.duplicated()
        df_thresholds=pd.DataFrame(columns=['Threshold','Percent of Categories'])
        for step in range(0,100,5):
            threshold=step/100
            count_above_threshold=(maxValuesObj / totals_df) >= threshold
            categories_with_match=maxValueIndexObj.loc[count_above_threshold]
            count_below_threshold=(maxValuesObj / totals_df) < threshold
            categories_without_match=maxValueIndexObj.loc[count_below_threshold]
            num_categories_with_match = len(categories_with_match)
            percent_categories_with_match = num_categories_with_match*100 / len(totals_df)
            print('for threshold {}, matched {} categories'.format(threshold, percent_categories_with_match))
            row={'Threshold':threshold,'Percent of Categories':percent_categories_with_match}
            df_thresholds=df_thresholds.append(row, ignore_index=True)
        df_thresholds.set_index('Threshold', inplace=True)
        return df_thresholds







