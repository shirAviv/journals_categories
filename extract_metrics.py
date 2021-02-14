from utils import Utils
from datetime import date,datetime,timedelta
import pickle
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
from visualization import Visualization
import numpy as np
from collections import Counter




path='D:\\shir\\study\\Bibliometrics\\journals'


class ExtractMetrics:
    def extract_wos_metrics(self, metrics_file):
        metrics_df=utils.load_csv_data_to_df(metrics_file)
        metrics_df['Full Journal Title']=metrics_df.apply(lambda row: pd.Series(row['Full Journal Title'].lower()), axis=1)
        journals_dict = utils.load_obj("wos_journals_dict")
        count_missing = 0
        journals_with_cats_metrics=pd.DataFrame(columns={'Journal name', 'Scopus Journal name', 'num categories', 'Total Cites', 'JIF', '5 year JIF', 'Eigenfactor', 'Norm Eigenfactor'})
        for journal_name, item in journals_dict.items():
            num_categories=len(item['categories'])
            scopus_journal_name=item['scopus_journal_name']

            if num_categories>5:
                print('num cats: {}, journal {}'.format(num_categories,journal_name))
            metrics=metrics_df[metrics_df['Full Journal Title']==journal_name].copy()
            if len(metrics)==0:
                # print("no metrics for {}".format(journal_name))
                count_missing+=1
                record={'Journal name':journal_name, 'Scopus Journal name': scopus_journal_name, 'num categories':num_categories, 'Total Cites':np.nan, 'JIF':np.nan, '5 year JIF':np.nan, 'Eigenfactor':np.nan, 'Norm Eigenfactor':np.nan}

            else:
                record={'Journal name':journal_name, 'Scopus Journal name':scopus_journal_name, 'num categories':num_categories, 'Total Cites':int(metrics['Total Cites'].values[0].replace(',','')), 'JIF':metrics['Journal Impact Factor'].values[0], '5 year JIF':metrics['5-Year Impact Factor'].values[0], 'Eigenfactor':float(metrics['Eigenfactor Score'].values[0]), 'Norm Eigenfactor':float(metrics['Normalized Eigenfactor'].values[0])}
            journals_with_cats_metrics=journals_with_cats_metrics.append(record, ignore_index=True)
        journals_with_cats_metrics['5 year JIF']=journals_with_cats_metrics['5 year JIF'].apply(pd.to_numeric, errors='coerce')
        journals_with_cats_metrics['JIF']=journals_with_cats_metrics['JIF'].apply(pd.to_numeric, errors='coerce')
        journals_with_cats_metrics['Eigenfactor'] = journals_with_cats_metrics['Eigenfactor'].apply(pd.to_numeric,
                                                                                                  errors='coerce')
        journals_with_cats_metrics['Norm Eigenfactor'] = journals_with_cats_metrics['Norm Eigenfactor'].apply(pd.to_numeric, errors='coerce')
        journals_with_cats_metrics['Total Cites'] = journals_with_cats_metrics['Total Cites'].apply(pd.to_numeric,
                                                                                                  errors='coerce')

        journals_with_cats_metrics['num categories']=journals_with_cats_metrics['num categories'].apply(pd.to_numeric, errors='coerce')


        print('missing {}'.format(count_missing))
        return journals_with_cats_metrics

    def extract_scopus_metrics(self, metrics_file):
        metrics_df=utils.load_csv_data_to_df(metrics_file, delimiter=';')
        metrics_df['Title']=metrics_df.apply(lambda row: pd.Series(row['Title'].lower()), axis=1)
        journals_dict = utils.load_obj("scopus_journals_dict")
        count_missing = 0
        journals_with_cats_metrics = pd.DataFrame(
            columns={'Journal name', 'num categories', 'Total Cites', 'Total Refs', 'SJR', 'H index', 'Categories Q',
                     'CiteScore'})
        for journal_name, item in journals_dict.items():
            num_categories = len(item['categories'])
            citeScore=item['CiteScore']
            sourceid=item['sourcerecord_id']
            issn=item['ISSN']
            eissn = item['eISSN']
            if num_categories > 50:
                print('num cats: {}, journal {}'.format(num_categories, journal_name))
            metrics = metrics_df[metrics_df['Title'] == journal_name].copy()
            if len(metrics) == 0:
                metrics = metrics_df[metrics_df['Sourceid'] == sourceid].copy()
                if len(metrics) == 0:

                    # print("no metrics for {}, issn {} source record id {}".format(journal_name, issn, sourceid))
                    count_missing += 1
                    record = {'Journal name': journal_name, 'num categories': num_categories, 'Total Cites': np.nan,
                          'Total Refs': np.nan, 'SJR': np.nan, 'H index':np.nan,'Categories Q': np.nan, 'CiteScore': citeScore}

            if len(metrics)!=0:
                    sjr = metrics['SJR'].values[0]
                    if isinstance(sjr,str):
                        sjr=sjr.replace(',','.')
                    record = {'Journal name': journal_name, 'num categories': num_categories,
                              'Total Cites': int(metrics['Total Cites (3years)'].values[0]),
                              'Total Refs': metrics['Total Refs.'].values[0],
                            'SJR': sjr,
                            'H index': metrics['H index'].values[0],
                            'Categories Q': metrics['Categories'].values[0],
                            'CiteScore': citeScore}
            journals_with_cats_metrics = journals_with_cats_metrics.append(record, ignore_index=True)
        journals_with_cats_metrics['Total Cites'] = journals_with_cats_metrics['Total Cites'].apply(pd.to_numeric,
                                                                                                  errors='coerce')
        journals_with_cats_metrics['Total Refs'] = journals_with_cats_metrics['Total Refs'].apply(pd.to_numeric, errors='coerce')
        journals_with_cats_metrics['SJR'] = journals_with_cats_metrics['SJR'].apply(pd.to_numeric,
                                                                                                    errors='coerce')
        journals_with_cats_metrics['H index'] = journals_with_cats_metrics['H index'].apply(
            pd.to_numeric, errors='coerce')
        journals_with_cats_metrics['CiteScore'] = journals_with_cats_metrics['CiteScore'].apply(pd.to_numeric,
                                                                                                    errors='coerce')

        journals_with_cats_metrics['num categories'] = journals_with_cats_metrics['num categories'].apply(pd.to_numeric,
                                                                                                          errors='coerce')

        print('missing {}'.format(count_missing))
        return journals_with_cats_metrics


    def get_wos_correlations(self, journals_with_cats_metrics):
        print('Calculating pearson correlation for Web Of Science metrics and num categories')

        df=journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['JIF'])==False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['JIF'].values)
        print('Pearsons correlation categories and JIF: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['5 year JIF']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['5 year JIF'].values)
        print('Pearsons correlation categories and 5 year JIF: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['Eigenfactor']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['Eigenfactor'].values)
        print('Pearsons correlation categories and Eigenfactor: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['Norm Eigenfactor']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['Norm Eigenfactor'].values)
        print('Pearsons correlation categories and Norm Eigenfactor: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['Total Cites']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['Total Cites'].values)
        print('Pearsons correlation categories and Total Cites: r {}. pValue {}'.format(r, pValue))


    def get_scopus_correlations(self, journals_with_cats_metrics):
        print('Calculating pearson correlation for Scopus metrics and num categories')
        df=journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['SJR'])==False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['SJR'].values)
        print('Pearsons correlation categories and SJR: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['CiteScore']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['CiteScore'].values)
        print('Pearsons correlation categories and 5 year CiteScore: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['H index']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['H index'].values)
        print('Pearsons correlation categories and H index: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['Total Cites']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['Total Cites'].values)
        print('Pearsons correlation categories and Total Cites: r {}. pValue {}'.format(r, pValue))

        df = journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['Total Refs']) == False]
        r, pValue = stats.pearsonr(df['num categories'].values, df['Total Refs'].values)
        print('Pearsons correlation categories and Total Refs: r {}. pValue {}'.format(r, pValue))

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


    def find_missing_journals_dont_use(self, journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus):
        journals_with_cats_metrics_wos=journals_with_cats_metrics_wos.sort_values(by='Journal name')
        journals_with_cats_metrics_scopus = journals_with_cats_metrics_scopus.sort_values(by='Journal name')

        missing_journals_idxs=set()
        for idx, item in journals_with_cats_metrics_scopus.iterrows():
            journal_name=item['Journal name']
            match=journals_with_cats_metrics_wos[journals_with_cats_metrics_wos['Scopus Journal name']==journal_name]
            if len(match)==0:
                missing_journals_idxs.add(idx)
                print(item)
        if len(missing_journals_idxs) > 0:
            journals_with_cats_metrics_scopus.drop(index=missing_journals_idxs, inplace=True)
        missing_journals_idxs = set()
        for idx, item in journals_with_cats_metrics_wos.iterrows():
            journal_name=item['Scopus Journal name']
            match=journals_with_cats_metrics_scopus[journals_with_cats_metrics_scopus['Journal name']==journal_name]
            if len(match)==0:
                missing_journals_idxs.add(idx)
                print(item)
        if len(missing_journals_idxs) > 0:
            journals_with_cats_metrics_wos.drop(index=missing_journals_idxs, inplace=True)
        dupes_df = journals_with_cats_metrics_wos.loc[journals_with_cats_metrics_wos.duplicated(['Scopus Journal name'], keep=False), :].copy()
        dupes_df.sort_values(by='Scopus Journal name', kind='mergesort', inplace=True)
        to_remove_wos_idx=[8241, 1410, 1411, 6743, 6811,  10970, 11804, 8718, 5459, 10315, 10320]
        journals_with_cats_metrics_wos.drop(index=to_remove_wos_idx, inplace=True)
        to_remove_scopus_idx = [10075, 4426, 4979]
        journals_with_cats_metrics_scopus.drop(index=to_remove_scopus_idx, inplace=True)
        journals_with_cats_metrics_wos.sort_values(by='Scopus Journal name', kind='mergesort', inplace=True)
        journals_with_cats_metrics_wos.reset_index(inplace=True, drop=True)
        journals_with_cats_metrics_scopus.reset_index(inplace=True, drop=True)

        return journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus


    def find_super_groups_and_intersection_all_journals_wos(self,df):
        categories=df['journals']
        sup_group_dict=dict()
        intersect_group_dict=dict()
        identity_group_dict=dict()
        sub_group_dict=dict()
        for category, journals_data in categories.items():

            journals = set(journals_data['Journal title'].unique())

            for compared_cat, compared_journals_data in categories.items():
                if category==compared_cat:
                    continue
                compared_journals = set(compared_journals_data['Journal title'].unique())
                if journals.isdisjoint(compared_journals):
                    continue
                if journals.issubset(compared_journals) and journals.issuperset(compared_journals):
                    if not category in identity_group_dict.keys():
                        identity_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'percent'])
                    journals_list = journals
                    percent = 100 * len(journals_list) / len(compared_journals)
                    record = {'category': compared_cat, 'journals': journals_list, 'percent': percent}
                    identity_group_dict[category] = identity_group_dict[category].append(record, ignore_index=True)
                    continue
                if journals.issubset(compared_journals):
                    if not category in sup_group_dict.keys():
                        sup_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'percent'])
                    journals_list=journals
                    percent=100*len(journals_list)/len(compared_journals)
                    record={'category':compared_cat,'journals':journals_list,'percent':percent}
                    sup_group_dict[category] = sup_group_dict[category].append(record, ignore_index=True)
                    continue
                if journals.issuperset(compared_journals):
                    if not category in sub_group_dict.keys():
                        sub_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'percent'])
                    journals_list=compared_journals
                    percent=100*len(journals_list)/len(journals)
                    record={'category':compared_cat,'journals':journals_list,'percent':percent}
                    sub_group_dict[category] = sub_group_dict[category].append(record, ignore_index=True)
                    continue
                if not category in intersect_group_dict.keys():
                    intersect_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'percent'])
                journals_list = journals.intersection(compared_journals)
                percent = 100 * len(journals_list) / len(journals)
                record = {'category': compared_cat, 'journals': journals_list, 'percent': percent}
                intersect_group_dict[category] = intersect_group_dict[category].append(record, ignore_index=True)

        return identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict

    def find_super_groups_and_intersection_all_journals(self,categories, threshold):
        sup_group_dict=dict()
        intersect_group_dict=dict()
        identity_group_dict=dict()
        sub_group_dict=dict()
        for category, journals_data in categories.items():
            if isinstance(journals_data, pd.DataFrame):
                journals=set(journals_data['Journal title'])
            else:
                journals = set(journals_data)

            for compared_cat, compared_journals_data in categories.items():
                if category==compared_cat:
                    continue
                if isinstance(compared_journals_data, pd.DataFrame):
                    compared_journals = set(compared_journals_data['Journal title'])
                else:
                    compared_journals = set(compared_journals_data)

                if journals.isdisjoint(compared_journals):
                    continue
                if journals.issubset(compared_journals) and journals.issuperset(compared_journals):
                    journals_list = journals
                    ratio = len(journals_list) / len(compared_journals)
                    if not category in identity_group_dict.keys():
                        identity_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio}
                    identity_group_dict[category] = identity_group_dict[category].append(record, ignore_index=True)
                    continue
                if journals.issubset(compared_journals):
                    journals_list = journals
                    ratio = len(journals_list) / len(compared_journals)
                    if ratio>=threshold:
                        # print('found in supset')
                        if not category in identity_group_dict.keys():
                            identity_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                        record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio}
                        identity_group_dict[category] = identity_group_dict[category].append(record, ignore_index=True)
                        continue
                    if not category in sup_group_dict.keys():
                        sup_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    record={'category':compared_cat,'journals':journals_list,'ratio':ratio}
                    sup_group_dict[category] = sup_group_dict[category].append(record, ignore_index=True)
                    continue
                if journals.issuperset(compared_journals):
                    journals_list = compared_journals
                    ratio = len(journals_list) / len(journals)
                    if ratio>=threshold:
                        # print('found in subset')
                        if not category in identity_group_dict.keys():
                            identity_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                        record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio}
                        identity_group_dict[category] = identity_group_dict[category].append(record, ignore_index=True)
                        continue
                    if not category in sub_group_dict.keys():
                        sub_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    record={'category':compared_cat,'journals':journals_list,'ratio':ratio}
                    sub_group_dict[category] = sub_group_dict[category].append(record, ignore_index=True)
                    continue
                journals_list = journals.intersection(compared_journals)
                ratio1 = len(journals_list) / len(journals)
                ratio2 = len(journals_list) / len(compared_journals)
                if ratio1 >= threshold and ratio2 >= threshold:
                    # print('found in intersect')
                    if not category in identity_group_dict.keys():
                        identity_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                    record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio1}
                    identity_group_dict[category] = identity_group_dict[category].append(record, ignore_index=True)
                    continue
                else:
                    if ratio1>= threshold:
                        if not category in sup_group_dict.keys():
                            sup_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                        record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio1}
                        sup_group_dict[category] = sup_group_dict[category].append(record, ignore_index=True)
                        continue
                    if ratio2>=threshold:
                        if not category in sub_group_dict.keys():
                            sub_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                        record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio2}
                        sub_group_dict[category] = sub_group_dict[category].append(record, ignore_index=True)
                        continue
                if not category in intersect_group_dict.keys():
                    intersect_group_dict[category] = pd.DataFrame(columns=['category', 'journals', 'ratio'])
                record = {'category': compared_cat, 'journals': journals_list, 'ratio': ratio1}
                intersect_group_dict[category] = intersect_group_dict[category].append(record, ignore_index=True)

        return identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict

    def num_journals_in_cat(self, e):
        return len(e)


    def find_groups(self,df):
        print('total num categories {}'.format(len(df)))

        categories=df['journals']

        for step in range(100,95, -5):
            threshold=step/100
            identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict = self.find_super_groups_and_intersection_all_journals(
                categories, threshold)
            print(
                '\nsize of identity group dict {}, super group dict {}, sub group dict {}, intersect group dict {}'.format(
                    len(identity_group_dict), len(sup_group_dict), len(sub_group_dict), len(intersect_group_dict)))
            if len(identity_group_dict) > 0:
                print('\nidentity groups for threshold {}'.format(threshold))
                for category, groups_data in identity_group_dict.items():
                    print(category, groups_data['category'].values, groups_data['ratio'].values)
            if len(sub_group_dict) > 0:
                print('\nsub groups for threshold {}'.format(threshold))
                for category, groups_data in sub_group_dict.items():
                    print(category, groups_data['category'].values, groups_data['ratio'].values)

            if len(sup_group_dict) > 0:
                print('\nsuper groups for threshold {}'.format(threshold))
                for category, groups_data in sup_group_dict.items():
                    print(category, groups_data['category'].values, groups_data['ratio'].values)
        intersect_categories=set(intersect_group_dict.keys())
        all_categories=set(categories.keys())
        disjoint_and_proper_subset_categories=all_categories.difference(intersect_categories)
        print(disjoint_and_proper_subset_categories)
        for cat in disjoint_and_proper_subset_categories:
            print('\ncat {}, num journals in cat {}'.format(cat, len(categories[cat])))
        intersect_df=pd.DataFrame(columns=['categories','num intersecting categories'])
        for key,val in intersect_group_dict.items():
            record={'categories':key,'num intersecting categories':int(len(val))}
            intersect_df=intersect_df.append(record, ignore_index=True)
        intersect_df['num intersecting categories']=intersect_df['num intersecting categories'].apply(pd.to_numeric, errors='coerce')

        return intersect_df, identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict

    def get_small_categories(self,df, name):
        df['count journals'] = df.apply(lambda row: pd.Series(len(row['journals'])), axis=1)
        sorted_cats = df.sort_values(by='count journals')
        small_cats=sorted_cats.loc[sorted_cats['count journals']<10, 'count journals']
        small_cats=small_cats.append(pd.Series(len(small_cats), index=[name]))
        print(small_cats)
        print('total {}'.format(len(small_cats)))
        return small_cats

    def get_large_categories(self,df, name):
        df['count journals'] = df.apply(lambda row: pd.Series(len(row['journals'])), axis=1)
        sorted_cats = df.sort_values(by='count journals')
        large_cats = sorted_cats.loc[sorted_cats['count journals'] > 300, 'count journals']
        large_cats = large_cats.append(pd.Series(len(large_cats), index=[name]))
        print(large_cats)
        print('total {}'.format(len(large_cats)))
        return large_cats


    def prep_data_for_venn_subset(self,dict):
        for cat, item in dict.items():
            num_journals_in_intersection = len(item['journals'].values[0])
            num_journals= num_journals_in_intersection / item['ratio'].values[0]
            subsets=(num_journals, 0, num_journals_in_intersection)
            labels=(cat, item['category'].values[0])
            vis.create_venn_diagrams(subsets, labels)
            print(num_journals)

    def prep_data_for_venn_intersect(self,dict, threshold,df):
        for cat, item in dict.items():
            num_journals_in_cat=len(df[cat]['journals'])
            records_above_threshold=item[item['ratio']>threshold]
            if len(records_above_threshold)==0:
                continue
            for record in records_above_threshold.iterrows():
                other_cat_name = record[1]['category']
                num_journals_in_intersection = len(record[1]['journals'])
                num_journals_in_other_cat = len(df[other_cat_name]['journals'])
                subsets = (num_journals_in_cat, num_journals_in_other_cat, num_journals_in_intersection)
                labels = (cat, other_cat_name)
                vis.create_venn_diagrams(subsets, labels)

    def run_small_and_large_cats_wos(self, df):
        name = 'Total_num_small_cats_wos'
        small_cats_wos=extractMetrics.get_small_categories(df, name)
        name = 'Total_num_large_cats_wos'
        large_cats_wos=extractMetrics.get_large_categories(df,name)
        return small_cats_wos, large_cats_wos

    def run_small_and_large_cats_scopus(self,df):
        name = 'Total_num_small_cats_scopus'
        small_cats_scopus=extractMetrics.get_small_categories(df.T,name)
        name = 'Total_num_large_cats_scopus'
        large_cats_scopus=extractMetrics.get_large_categories(df.T,name)
        return small_cats_scopus, large_cats_scopus

    def run_small_and_large_cats(self,wos_df, scopus_df):
        small_cats_wos, large_cats_wos=self.run_small_and_large_cats_wos(wos_df)
        small_cats_scopus, large_cats_scopus=self.run_small_and_large_cats_scopus(scopus_df)
        small_cats_wos=small_cats_wos.append(small_cats_scopus)
        utils.save_obj(small_cats_wos,"inter_systems_small_categories")
        utils.write_to_csv(small_cats_wos, 'inter_systems_small_categories.csv', index=True)
        large_cats_wos=large_cats_wos.append(large_cats_scopus)
        utils.save_obj(large_cats_wos,"inter_systems_large_categories")
        utils.write_to_csv(large_cats_wos,'inter_systems_large_categories.csv', index=True)

    def prep_data_for_venn_plots(self,wos_df,sub_group_dict_wos, intersect_group_dict_wos, scopus_df, sub_group_dict_scopus,intersect_group_dict_scopus ):
        extractMetrics.prep_data_for_venn_subset(sub_group_dict_wos)
        extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.9, wos_df.T)
        print('0.8')
        extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.8, wos_df.T)

        extractMetrics.prep_data_for_venn_subset(sub_group_dict_scopus)
        extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_scopus,0.8,scopus_df)

    def plt_histograms_intersect(self):
        intersect_df1= utils.load_obj('wos_num_intersections')
        vis.plt_histogram_cats_intersection(intersect_df1, title='Categories intersections in WOS')
        intersect_df2= utils.load_obj('scopus_num_intersections')
        vis.plt_histogram_cats_intersection(intersect_df2, title='Categories intersections in Scopus')

    def plt_histograms_num_cats(self):
        journals_with_cats_metrics_wos = utils.load_obj("wos_journals_with_metrics")
        vis.plt_histogram_cats(journals_with_cats_metrics_wos, title="WOS number of categories per journal distribution")
        journals_with_cats_metrics_scopus = utils.load_obj("scopus_journals_with_metrics")
        vis.plt_histogram_cats(journals_with_cats_metrics_scopus, title="Scopus number of categories per journal distribution")

    def create_journals_with_cats_metrics(self):
        journals_with_cats_metrics=extractMetrics.extract_wos_metrics('wos_journals_metrics.csv')
        utils.save_obj(journals_with_cats_metrics,"wos_journals_with_metrics")

        journals_with_cats_metrics=extractMetrics.extract_scopus_metrics('scopus_scores_2019.csv')
        utils.save_obj(journals_with_cats_metrics, "scopus_journals_with_metrics")

        journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus= extractMetrics.find_missing_journals(journals_with_cats_metrics_wos,journals_with_cats_metrics_scopus)
        utils.save_obj(journals_with_cats_metrics_wos,"wos_journals_with_metrics")
        utils.save_obj(journals_with_cats_metrics_scopus, "scopus_journals_with_metrics")

    def get_correlations_all_journals(self):
        journals_with_cats_metrics_wos = utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_scopus = utils.load_obj("scopus_journals_with_metrics")
        self.get_wos_correlations(journals_with_cats_metrics_wos)
        self.get_scopus_correlations(journals_with_cats_metrics_scopus)
        self.get_inter_system_correlations(journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus)


    def create_journal_ranking_by_category_wos(self,df, journals_with_metrics):
        categories = df['journals']
        for category, journals_data in categories.items():
            count_missing_metrics=0
            num_journals=len(journals_data)
            jIF=[]
            eigenfactor=[]
            norm_Eigenfactor=[]
            five_year_JIF=[]
            for journal in journals_data.iterrows():
                journal_name=journal[1]['Journal title']
                metrics = journals_with_metrics[journals_with_metrics['Journal name'] == journal_name].copy()
                if len(metrics) == 0:
                    count_missing_metrics+=1
                    jIF.append(np.nan)
                    five_year_JIF.append(np.nan)
                    eigenfactor.append(np.nan)
                    norm_Eigenfactor.append(np.nan)
                else:
                    jIF.append(metrics['JIF'].values[0])
                    five_year_JIF.append(metrics['5 year JIF'].values[0])
                    eigenfactor.append(metrics['Eigenfactor'].values[0])
                    norm_Eigenfactor.append(metrics['Norm Eigenfactor'].values[0])

            journals_data['JIF']=jIF
            journals_data['Five year JIF'] = five_year_JIF
            journals_data['Eigenfactor'] = eigenfactor
            journals_data['Norm Eigenfactor'] = norm_Eigenfactor

            journals_data.sort_values(by=['JIF'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Rank JIF'] = 4
            journals_data.loc[0:num_journals/4, 'Rank JIF'] = 1
            journals_data.loc[(num_journals / 4):num_journals/2, 'Rank JIF'] = 2
            journals_data.loc[(num_journals / 2):3*(num_journals/4), 'Rank JIF'] = 3
            del (journals_data['index'])

            journals_data.sort_values(by=['Five year JIF'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Rank Five year JIF'] = 4
            journals_data.loc[0:num_journals / 4, 'Rank Five year JIF'] = 1
            journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Five year JIF'] = 2
            journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Five year JIF'] = 3
            del (journals_data['index'])

            journals_data.sort_values(by=['Eigenfactor'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Rank Eigenfactor'] = 4
            journals_data.loc[0:num_journals / 4, 'Rank Eigenfactor'] = 1
            journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Eigenfactor'] = 2
            journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Eigenfactor'] = 3
            del (journals_data['index'])

            journals_data.sort_values(by=['Norm Eigenfactor'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Rank Norm Eigenfactor'] = 4
            journals_data.loc[0:num_journals / 4, 'Rank Norm Eigenfactor'] = 1
            journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Norm Eigenfactor'] = 2
            journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Norm Eigenfactor'] = 3
            del (journals_data['index'])
            if (count_missing_metrics/num_journals>0.1):
                print('in cat {} number of journals with missing metrics {} out of {}'.format(category,count_missing_metrics,num_journals))
        return categories

    def get_categories_ranking_mismatch(self):
        journals_with_cats_metrics_wos = utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_wos['Rank JIF'] = np.nan
        journals_with_cats_metrics_wos['Rank Five year JIF'] = np.nan
        journals_with_cats_metrics_wos['Rank Eigenfactor'] = np.nan
        journals_with_cats_metrics_wos['Rank Norm Eigenfactor'] = np.nan
        categories_with_ranks_df=utils.load_obj('categories_with_ranks_df')
        wos_journals_dict = utils.load_obj("wos_journals_dict")
        for journal_name, item in wos_journals_dict.items():
            rank_jIF = []
            rank_eigenfactor = []
            rank_norm_eigenfactor = []
            rank_five_year_JIF = []
            for category in item['categories']:
                journals_in_cat= categories_with_ranks_df[category]
                metrics = journals_in_cat[journals_in_cat['Journal name'] == journal_name].copy()
                if len(metrics) == 0:
                    rank_jIF.append(np.nan)
                    rank_five_year_JIF.append(np.nan)
                    rank_eigenfactor.append(np.nan)
                    rank_norm_eigenfactor.append(np.nan)
                else:
                    rank_jIF.append(metrics['Rank JIF'].values[0])
                    rank_five_year_JIF.append(metrics['Rank five year JIF'].values[0])
                    rank_eigenfactor.append(metrics['Rank Eigenfactor'].values[0])
                    rank_norm_eigenfactor.append(metrics['Rank Norm Eigenfactor'].values[0])







    def calc_mismatch(self, ranks_list):
        ret_val=0
        if len(list)>1:
            occurence_count = Counter(ranks_list)
            most_frequent = occurence_count.most_common(1)[0][0]
            for item in ranks_list:
                if np.isnan(item):
                    return np.nan
                ret_val+=abs(most_frequent-item)
        return ret_val


if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    vis=Visualization()
    extractMetrics=ExtractMetrics()


    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
    # intersect_df1, identity_group_dict_wos, sup_group_dict_wos, sub_group_dict_wos, intersect_group_dict_wos=extractMetrics.find_groups(df1)
    # utils.save_obj(intersect_df1,'wos_num_intersections')

    # identity_group_dict, sup_group_dict, sub_group_dict, intersect_group_dict=extractMetrics.find_super_groups_and_intersection_all_journals_wos(df1)
    df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
    # intersect_df2, identity_group_dict_scopus, sup_group_dict_scopus, sub_group_dict_scopus, intersect_group_dict_scopus=extractMetrics.find_groups(df2.T)
    # utils.save_obj(intersect_df2, 'scopus_num_intersections')

    # extractMetrics.get_correlations_all_journals()
    # extractMetrics.run_small_and_large_cats(df1,df2)
    # extractMetrics.prep_data_for_venn_plots(df1, sub_group_dict_wos,intersect_group_dict_wos, df2, sub_group_dict_scopus, intersect_group_dict_scopus)

    # journals_with_cats_metrics_wos = utils.load_obj("wos_journals_with_metrics")
    # categories_with_ranks_df=extractMetrics.create_journal_ranking_by_category_wos(df1,journals_with_cats_metrics_wos)
    # utils.save_obj(categories_with_ranks_df,'categories_with_ranks_df')
    extractMetrics.get_categories_ranking_mismatch()


