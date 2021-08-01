from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from utils import Utils
from datetime import datetime
from visualization import Visualization
import networkx as nx



class ExtractMetricsIntra:
    def extract_wos_metrics(self, metrics_df, journals_dict):

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

    def extract_scopus_metrics(self, metrics_df, journals_dict):

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



    def calc_correlations_gaps_in_percentiles_by_categories(self, journals_with_metrics, db, column):
        print('Calculating pearson correlation for {} metrics differences within categories and num categories'.format(db))

        df = journals_with_metrics[np.isnan(journals_with_metrics[column])==False]
        r, pValue = stats.pearsonr(df['num categories'].values, df[column].values)
        print('Pearsons correlation categories and {} within category: r {}. pValue {}'.format(column, r, pValue))



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
        idx = intersect_df['num intersecting categories'].argmax()
        largest_intersecting_category=intersect_df.T[idx]['categories']
        num_largest_intersecting_category = intersect_df.T[idx]['num intersecting categories']
        print('largest intersecting category {}, num intersecting categories {}'.format(largest_intersecting_category, num_largest_intersecting_category))
        # for category in intersect_group_dict.items()
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
        large_cats = sorted_cats.loc[sorted_cats['count journals'] > 200, 'count journals']
        large_cats = large_cats.append(pd.Series(len(large_cats), index=[name]))
        print(large_cats)
        print('total {}'.format(len(large_cats)))
        return large_cats

    def get_num_journals_in_cats_distribution(self, df,name):
        df['count journals'] = df.apply(lambda row: pd.Series(len(row['journals'])), axis=1)
        sorted_cats = df.sort_values(by='count journals')
        interval=pd.interval_range(df['count journals'].min(), df['count journals'].max(), periods=15)
        interval=pd.interval_range(start=df['count journals'].min(), periods=30, freq=15)

        df['myQuanBins'] = pd.cut(df['count journals'], bins=interval)

        # df1=df.groupby(by='count journals').count().journals
        # bins=pd.qcut(df.groupby(by='count journals').count(), 10)
        df_sorted=df['myQuanBins'].value_counts().sort_index()
        self.vis.plt_histogram_journals(df=df_sorted, title= name)


    def prep_data_for_venn_subset(self,dict):
        fig, ax= self.vis.get_subplots_for_venn()
        x=0
        y=0
        for cat, item in dict.items():
            if len(item)>1 and len(item)<3:
                # continue
                num_journals_in_intersection1 = len(item[:1]['journals'].values[0]).__round__()
                num_journals_in_intersection2 = len(item[1:2]['journals'].values[0]).__round__()
                num_journals = (num_journals_in_intersection1 / item[:1]['ratio'].values[0]-num_journals_in_intersection2).__round__()
                # num_journals2 = num_journals_in_intersection2 / item[1]['ratio'].values[0]-num_journals_in_intersection1
                subsets = (num_journals, 0, num_journals_in_intersection2, 0,num_journals_in_intersection1,0,0)
                idx = cat.find(' ', cat.find(' ') + 1)
                cat_name = cat
                if idx > -1:
                    cat_name = cat_name[:idx] + '\n' + cat_name[idx + 1:]
                second_cat = item[1:2]['category'].values[0]
                idx = second_cat.find(' ', second_cat.find(' ') + 1)
                if idx > -1:
                    second_cat = second_cat[:idx] + '\n' + second_cat[idx + 1:]
                third_cat = item[:1]['category'].values[0]
                idx = third_cat.find(' ', third_cat.find(' ') + 1)
                if idx > -1:
                    third_cat = third_cat[:idx] + '\n' + third_cat[idx + 1:]
                labels = (cat_name, second_cat , third_cat )
                self.vis.create_venn_diagrams(subsets, labels, ax[x][y], True)
                y += 1
            if len(item)==1:
                # continue;
                num_journals_in_intersection = len(item['journals'].values[0]).__round__()
                num_journals= (num_journals_in_intersection / item['ratio'].values[0]).__round__()
                subsets=(num_journals, 0, num_journals_in_intersection)
                idx = cat.find(' ', cat.find(' ') + 1)
                cat_name = cat
                if idx > -1:
                    cat_name = cat_name[:idx] + '\n' + cat_name[idx + 1:]
                    idx = cat_name.find(' ', cat_name.find(' ', idx+1) + 1)
                    if idx>-1:
                        cat_name = cat_name[:idx] + '\n' + cat_name[idx + 1:]
                second_cat = item['category'].values[0]
                idx = second_cat.find(' ', second_cat.find(' ') + 1)
                if idx > -1:
                    second_cat = second_cat[:idx] + '\n' + second_cat[idx + 1:]
                labels=(cat_name,second_cat )
                self.vis.create_venn_diagrams(subsets, labels, ax[x][y])
                print(num_journals)
                y+=1
                if y==4:
                    x+=1
                    y=0
        self.vis.plt_show_and_title('')

    def prep_data_for_venn_intersect(self,dict, threshold,df_from, df_to):
        fig, ax = self.vis.get_subplots_for_venn()
        x = 0
        y = 0
        for cat, item in dict.items():
            if cat.endswith('ities)'):
                continue
            if cat.startswith('Ling'):
                continue
            num_journals_in_cat=len(df_from[cat]['journals'])
            records_above_threshold=item[item['ratio']>threshold]
            records_above_threshold=records_above_threshold[records_above_threshold['ratio']<(threshold+0.05)]
            if len(records_above_threshold)==0:
                continue
            else:
                print(cat)
            for record in records_above_threshold.iterrows():
                other_cat_name = record[1]['category']
                num_journals_in_intersection = len(record[1]['journals'])
                num_journals_in_other_cat = len(df_to[other_cat_name]['journals'])
                subsets = (num_journals_in_cat-num_journals_in_intersection, num_journals_in_other_cat-num_journals_in_intersection, num_journals_in_intersection)
                idx=cat.find(' ',cat.find(' ')+1)
                cat_name = cat
                if idx>-1:
                    cat_name=cat_name[:idx]+'\n'+cat_name[idx+1:]
                idx = other_cat_name.find(' ', other_cat_name.find(' ') + 1)
                if idx>-1:
                    other_cat_name = other_cat_name[:idx] + '\n' + other_cat_name[idx + 1:]
                labels = (cat_name, other_cat_name)
                self.vis.create_venn_diagrams(subsets, labels, ax[x][y])
                # print(cat)
                # print(other_cat_name)
                y += 1
                if y == 3:
                    x += 1
                    y = 0
        self.vis.plt_show_and_title('')


    def run_small_and_large_cats_wos(self, df):
        name = 'Total_num_small_cats_wos'
        small_cats_wos=self.get_small_categories(df, name)
        name = 'Total_num_large_cats_wos'
        large_cats_wos=self.get_large_categories(df, name)
        return small_cats_wos, large_cats_wos

    def run_small_and_large_cats_scopus(self,df):
        name = 'Total_num_small_cats_scopus'
        small_cats_scopus=self.get_small_categories(df, name)
        name = 'Total_num_large_cats_scopus'
        large_cats_scopus=self.get_large_categories(df, name)
        return small_cats_scopus, large_cats_scopus

    def run_small_and_large_cats(self,wos_df, scopus_df):
        small_cats_wos, large_cats_wos=self.run_small_and_large_cats_wos(wos_df)
        small_cats_scopus, large_cats_scopus=self.run_small_and_large_cats_scopus(scopus_df)
        small_cats_wos=small_cats_wos.append(small_cats_scopus)
        self.utils.save_obj(small_cats_wos,"intra_systems_small_categories")
        self.utils.write_to_csv(small_cats_wos, 'intra_systems_small_categories.csv', index=True)
        large_cats_wos=large_cats_wos.append(large_cats_scopus)
        self.utils.save_obj(large_cats_wos,"intra_systems_large_categories")
        self.utils.write_to_csv(large_cats_wos,'intra_systems_large_categories.csv', index=True)

    def prep_data_for_venn_plots(self,wos_df,sub_group_dict_wos, intersect_group_dict_wos, scopus_df, sub_group_dict_scopus,intersect_group_dict_scopus ):
        # extractMetrics.prep_data_for_venn_subset(sub_group_dict_wos)
        self.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.85, wos_df.T)
        # print('0.8')
        # extractMetrics.prep_data_for_venn_intersect(intersect_group_dict_wos, 0.8, wos_df.T)

        # self.prep_data_for_venn_subset(sub_group_dict_scopus)
        # self.prep_data_for_venn_intersect(intersect_group_dict_scopus, 0.80, scopus_df)

    def plt_histograms_intersect(self):
        intersect_df1= self.utils.load_obj('wos_num_intersections')
        self.vis.plt_histogram_cats_intersection(intersect_df1, title='Categories intersections in WOS')
        intersect_df2= self.utils.load_obj('scopus_num_intersections')
        self.vis.plt_histogram_cats_intersection(intersect_df2, title='Categories intersections in Scopus')

    def plt_histograms_num_cats(self):
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        self.vis.plt_histogram_cats(journals_with_cats_metrics_wos, title="WOS number of categories per journal distribution")
        journals_with_cats_metrics_scopus = self.utils.load_obj("scopus_journals_with_metrics")
        self.vis.plt_histogram_cats(journals_with_cats_metrics_scopus, title="Scopus number of categories per journal distribution")

    def create_journals_with_cats_metrics(self,metrics_file_wos, journals_dict_wos, metrics_file_scopus, journals_dict_scopus):
        metrics_df = self.utils.load_csv_data_to_df(metrics_file_wos)
        metrics_df['Full Journal Title'] = metrics_df.apply(lambda row: pd.Series(row['Full Journal Title'].lower()),
                                                            axis=1)
        journals_with_cats_metrics_wos=self.extract_wos_metrics(metrics_df, journals_dict_wos)
        self.utils.save_obj(journals_with_cats_metrics_wos,"wos_journals_with_metrics_v3")

        metrics_df = self.utils.load_csv_data_to_df(metrics_file_scopus, delimiter=';')
        metrics_df['Title'] = metrics_df.apply(lambda row: pd.Series(row['Title'].lower()), axis=1)

        journals_with_cats_metrics_scopus=self.extract_scopus_metrics(metrics_df,journals_dict_scopus)
        self.utils.save_obj(journals_with_cats_metrics_scopus, "scopus_journals_with_metrics_v2")

        journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus= self.find_missing_journals_dont_use(journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus)
        self.utils.save_obj(journals_with_cats_metrics_wos,"wos_journals_with_metrics_v3")
        self.utils.save_obj(journals_with_cats_metrics_scopus, "scopus_journals_with_metrics_v2")

    def get_correlations_all_journals(self):
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_scopus = self.utils.load_obj("scopus_journals_with_metrics")
        self.get_wos_correlations(journals_with_cats_metrics_wos)
        self.get_scopus_correlations(journals_with_cats_metrics_scopus)
        # self.get_inter_system_correlations(journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus)


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
            journals_data['Percentile Rank JIF'] = journals_data.JIF.rank(pct=True)
            # journals_data.loc[0:num_journals/4, 'Rank JIF'] = 1
            # journals_data.loc[(num_journals / 4):num_journals/2, 'Rank JIF'] = 2
            # journals_data.loc[(num_journals / 2):3*(num_journals/4), 'Rank JIF'] = 3
            del (journals_data['index'])

            # journals_data.sort_values(by=['Five year JIF'], inplace=True, ascending=False)
            # journals_data.reset_index(inplace=True)
            # journals_data['Rank Five year JIF'] = 4
            # journals_data.loc[0:num_journals / 4, 'Rank Five year JIF'] = 1
            # journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Five year JIF'] = 2
            # journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Five year JIF'] = 3
            # del (journals_data['index'])

            # journals_data.sort_values(by=['Eigenfactor'], inplace=True, ascending=False)
            # journals_data.reset_index(inplace=True)
            # journals_data['Rank Eigenfactor'] = 4
            # journals_data.loc[0:num_journals / 4, 'Rank Eigenfactor'] = 1
            # journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Eigenfactor'] = 2
            # journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Eigenfactor'] = 3
            # del (journals_data['index'])

            journals_data.sort_values(by=['Norm Eigenfactor'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Percentile Rank Norm Eigenfactor'] = journals_data['Norm Eigenfactor'].rank(pct=True)
            # journals_data.loc[0:num_journals / 4, 'Rank Norm Eigenfactor'] = 1
            # journals_data.loc[(num_journals / 4):num_journals / 2, 'Rank Norm Eigenfactor'] = 2
            # journals_data.loc[(num_journals / 2):3 * (num_journals / 4), 'Rank Norm Eigenfactor'] = 3
            del (journals_data['index'])
            if (count_missing_metrics/num_journals>0.1):
                print('in cat {} number of journals with missing metrics {} out of {}'.format(category,count_missing_metrics,num_journals))
        return categories

    def create_journal_ranking_by_category_scopus(self,df, journals_with_metrics):
        categories = df['journals']
        for category, journals_list in categories.items():
            journals_data=pd.DataFrame(columns=['Journal title', 'SJR','CiteScore', 'Percentile Rank SJR', 'Percentile Rank CiteScore'])
            journals_data['Journal title']=journals_list
            count_missing_metrics=0
            num_journals=len(journals_list)
            sJR=[]
            citeScore=[]
            for journal in journals_data.iterrows():
                journal_name=journal[1]['Journal title']
                metrics = journals_with_metrics[journals_with_metrics['Journal name'] == journal_name].copy()
                if len(metrics) == 0:
                    count_missing_metrics+=1
                    sJR.append(np.nan)
                    citeScore.append(np.nan)
                else:
                    sJR.append(metrics['SJR'].values[0])
                    citeScore.append(metrics['CiteScore'].values[0])

            journals_data['SJR']=sJR
            journals_data['CiteScore'] = citeScore

            journals_data.sort_values(by=['SJR'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Percentile Rank SJR'] = journals_data.SJR.rank(pct=True)

            del (journals_data['index'])

            journals_data.sort_values(by=['CiteScore'], inplace=True, ascending=False)
            journals_data.reset_index(inplace=True)
            journals_data['Percentile Rank CiteScore'] = journals_data['CiteScore'].rank(pct=True)

            del (journals_data['index'])
            if (count_missing_metrics/num_journals>0.1):
                print('in cat {} number of journals with missing metrics {} out of {}'.format(category,count_missing_metrics,num_journals))
            categories[category]=journals_data
        return categories


    def create_journal_ranking_by_categor(self, df_wos, df_scopus):
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        categories_with_ranks_df_wos = self.create_journal_ranking_by_category_wos(df_wos,
                                                                                                  journals_with_cats_metrics_wos)
        self.utils.save_obj(categories_with_ranks_df_wos, 'categories_with_ranks_df_wos')

        journals_with_cats_metrics_scopus = self.utils.load_obj("scopus_journals_with_metrics")
        categories_with_ranks_df_scopus = self.create_journal_ranking_by_category_scopus(df_scopus,
                                                                                                        journals_with_cats_metrics_scopus)
        self.utils.save_obj(categories_with_ranks_df_scopus, 'categories_with_ranks_df_scopus')

    def get_categories_ranking_mismatch_wos(self):
        journals_with_cats_metrics_wos = self.utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_wos['Percentile ranking Max Min Range'] = np.nan
        journals_with_cats_metrics_wos['Percentile ranking ASOS'] = np.nan
        journals_with_cats_metrics_wos['Percentile ranking var'] = np.nan
        journals_with_cats_metrics_wos['Percentile ranking mean'] = np.nan


        # journals_with_cats_metrics_wos['Rank Five year JIF'] = np.nan
        # journals_with_cats_metrics_wos['Rank Eigenfactor'] = np.nan
        # journals_with_cats_metrics_wos['Rank Norm Eigenfactor'] = np.nan
        categories_with_ranks_df=self.utils.load_obj('categories_with_ranks_df_wos')
        wos_journals_dict = self.utils.load_obj("wos_journals_dict")
        for journal_name, item in wos_journals_dict.items():
            rank_jIF = []
            rank_eigenfactor = []
            rank_norm_eigenfactor = []
            rank_five_year_JIF = []
            if len(item['categories'])==1:
                continue
            for category in item['categories']:
                journals_in_cat= categories_with_ranks_df[category]
                metrics = journals_in_cat[journals_in_cat['Journal title'] == journal_name].copy()
                if len(metrics) == 0:
                    rank_jIF.append(np.nan)
                    # rank_five_year_JIF.append(np.nan)
                    # rank_eigenfactor.append(np.nan)
                    # rank_norm_eigenfactor.append(np.nan)
                else:
                    rank_jIF.append(metrics['Percentile Rank JIF'].values[0]*100)
                    # rank_five_year_JIF.append(metrics['Rank five year JIF'].values[0])
                    # rank_eigenfactor.append(metrics['Rank Eigenfactor'].values[0])
                    # rank_norm_eigenfactor.append(metrics['Rank Norm Eigenfactor'].values[0])
            rank_jIF=np.array(rank_jIF)
            max_min_range=rank_jIF.ptp()

            journals_with_cats_metrics_wos.loc[
                (journals_with_cats_metrics_wos['Journal name'] == journal_name), 'Percentile ranking Max Min Range'] = max_min_range
            avg_sum_of_squares=np.linalg.norm(rank_jIF)/np.size(rank_jIF)
            journals_with_cats_metrics_wos.loc[
                (journals_with_cats_metrics_wos['Journal name'] == journal_name), 'Percentile ranking ASOS'] = avg_sum_of_squares
            var_JIF =np.var(rank_jIF)
            journals_with_cats_metrics_wos.loc[
                (journals_with_cats_metrics_wos['Journal name'] == journal_name), 'Percentile ranking var'] = var_JIF
            mean_JIF =np.mean(rank_jIF)
            journals_with_cats_metrics_wos.loc[
                (journals_with_cats_metrics_wos['Journal name'] == journal_name), 'Percentile ranking mean'] = mean_JIF
            # asos2_JIF = (np.sum((rank_jIF-mean_JIF)**2))/np.size(rank_jIF)
            # journals_with_cats_metrics_wos.loc[
            #     (journals_with_cats_metrics_wos['Journal name'] == journal_name), 'JIF ASOS2'] = asos2_JIF

        return journals_with_cats_metrics_wos

    def get_categories_ranking_mismatch_scopus(self):
        journals_with_cats_metrics = self.utils.load_obj("scopus_journals_with_metrics")
        journals_with_cats_metrics['Percentile ranking Max Min Range'] = np.nan
        journals_with_cats_metrics['Percentile ranking ASOS'] = np.nan
        journals_with_cats_metrics['Percentile ranking var'] = np.nan
        journals_with_cats_metrics['Percentile ranking mean'] = np.nan

        categories_with_ranks_df=self.utils.load_obj('categories_with_ranks_df_scopus')
        journals_dict = self.utils.load_obj("scopus_journals_dict")
        for journal_name, item in journals_dict.items():
            rank_sJR = []
            rank_citeScore = []
            if len(item['categories'])==1:
                continue
            for category in item['categories']:
                journals_in_cat= categories_with_ranks_df[category]
                metrics = journals_in_cat[journals_in_cat['Journal title'] == journal_name].copy()
                if len(metrics) == 0:
                    rank_sJR.append(np.nan)
                    # rank_citeScore.append(np.nan)
                else:
                    rank_sJR.append(metrics['Percentile Rank SJR'].values[0]*100)
                    # rank_citeScore.append(metrics['Rank Eigenfactor'].values[0])
            rank_sJR=np.array(rank_sJR)
            max_min_range=rank_sJR.ptp()
            journals_with_cats_metrics.loc[
                (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking Max Min Range'] = max_min_range
            avg_sum_of_squares=np.linalg.norm(rank_sJR)/np.size(rank_sJR)
            journals_with_cats_metrics.loc[
                (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking ASOS'] = avg_sum_of_squares
            var_SJR = np.var(rank_sJR)
            journals_with_cats_metrics.loc[
                (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking var'] = var_SJR
            mean_SJR = np.mean(rank_sJR)
            journals_with_cats_metrics.loc[
                (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking mean'] = mean_SJR

        return journals_with_cats_metrics

    def get_categories_ranking_mismatch(self):
        journals_with_cats_metrics_ranking_wos=self.get_categories_ranking_mismatch_wos()
        self.utils.save_obj(journals_with_cats_metrics_ranking_wos,'journals_with_cats_metrics_wos')
        journals_with_cats_metrics_ranking_scopus=self.get_categories_ranking_mismatch_scopus()
        self.utils.save_obj(journals_with_cats_metrics_ranking_scopus,'journals_with_cats_metrics_scopus')


    def analyse_categories_ranking_mismatch(self):
        journals_with_cats_metrics_ranking_wos = self.utils.load_obj('journals_with_cats_metrics_wos')
        journals_with_multiple_categories=journals_with_cats_metrics_ranking_wos.loc[journals_with_cats_metrics_ranking_wos['num categories']>1]
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='WOS', column='Percentile ranking Max Min Range')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='WOS', column='Percentile ranking ASOS')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='WOS', column='Percentile ranking var')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='WOS', column='Percentile ranking mean')


        # journals_with_multiple_categories=journals_with_multiple_categories.loc[np.isnan(journals_with_multiple_categories['JIF Max Min Range'])==False]
        journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_scopus')

        # journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_ranking_scopus')
        journals_with_multiple_categories=journals_with_cats_metrics_ranking_scopus.loc[journals_with_cats_metrics_ranking_scopus['num categories']>1]
        # and journals_with_cats_metrics_ranking_scopus['num categories']<10]
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='Scopus',
                                                                 column='Percentile ranking Max Min Range')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='Scopus', column='Percentile ranking ASOS')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='Scopus', column='Percentile ranking var')
        self.calc_correlations_gaps_in_percentiles_by_categories(journals_with_multiple_categories, db='Scopus', column='Percentile ranking mean')



    def create_clusters_by_categories(self, df_wos, df_scopus):
        journals_in_small_cats = self.get_outlier_journals(df_wos)
        self.create_clusters_by_categories_wos(remove_outliers=True, outlier_journals=journals_in_small_cats)
        journals_in_small_cats = self.get_outlier_journals(df_scopus, wos=False)
        self.create_clusters_by_categories_scopus(remove_outliers=True, outlier_journals=journals_in_small_cats)

    def create_clusters_by_categories_wos(self, remove_outliers=False, outlier_journals=None):
        journals_with_cats_metrics = self.utils.load_obj("wos_journals_with_metrics")
        journals_with_cats_metrics_no_outlier_JIF=journals_with_cats_metrics[journals_with_cats_metrics.JIF<=100]
        if remove_outliers:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_JIF[~journals_with_cats_metrics_no_outlier_JIF['Journal name'].isin(outlier_journals)]
        else:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_JIF
        groups_by_num_categories=journals_with_cats_metrics_for_analysis.groupby('num categories')
        plt_by='JIF'
        title = 'JIF statistics by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_cats_metrics_for_analysis, x='num categories', plt_by=plt_by, title=title)
        self.vis.plt_clear()
        title = 'Highest JIF by number of categories'
        self.vis.plt_groups_max(groups_by_num_categories, plt_by=plt_by, title=title)
        self.vis.plt_clear()


    def create_clusters_by_categories_scopus(self, remove_outliers=False, outlier_journals=None):
        # self.vis.plt_group_date_with_box_plot(groups_by_num_categories, plt_by=plt_by, title=title)
        journals_with_cats_metrics = self.utils.load_obj("scopus_journals_with_metrics")
        journals_with_cats_metrics_no_outlier_citescore = journals_with_cats_metrics[
            journals_with_cats_metrics.CiteScore <= 200]
        if remove_outliers:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_citescore[~journals_with_cats_metrics_no_outlier_citescore['Journal name'].isin(outlier_journals)]
        else:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_citescore
        groups_by_num_categories = journals_with_cats_metrics_for_analysis.groupby('num categories')

        journals_with_cats_metrics_no_outlier_sjr = journals_with_cats_metrics[
            journals_with_cats_metrics['num categories'] <= 10]
        # journals_with_cats_metrics_no_outlier_sjr = journals_with_cats_metrics[
        # journals_with_cats_metrics.SJR <= 70]
        if remove_outliers:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_sjr[~journals_with_cats_metrics_no_outlier_sjr['Journal name'].isin(outlier_journals)]
        else:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_sjr
        groups_by_num_categories = journals_with_cats_metrics_for_analysis.groupby('num categories')
        plt_by = 'SJR'
        title = 'SJR statistics by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_cats_metrics_for_analysis, x='num categories',
                                         plt_by=plt_by, title=title)
        self.vis.plt_clear()
        # self.vis.plt_groups_data(groups_by_num_categories, plt_by=plt_by, title=title)
        title = 'Highest SJR by number of categories'
        self.vis.plt_groups_max(groups_by_num_categories, plt_by=plt_by, title=title)
        self.vis.plt_clear()

    def plot_min_max_by_cats(self):
        journals_with_cats_metrics_ranking_wos = self.utils.load_obj('journals_with_cats_metrics_wos')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking Max Min Range'
        title = 'WOS - MM of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')
        journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_scopus')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
            journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        plt_by = 'Percentile ranking Max Min Range'
        title = 'Scopus - MM of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')


    def plot_avg_sum_of_squares_by_cats(self):
        journals_with_cats_metrics_ranking_wos = self.utils.load_obj('journals_with_cats_metrics_wos')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking ASOS'
        title = 'WOS - ASOS of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')
        journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_scopus')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
            journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        plt_by = 'Percentile ranking ASOS'
        title = 'Scopus - ASOS of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')


    def plot_mean_variance_by_cats(self ):
        journals_with_cats_metrics_ranking_wos = self.utils.load_obj('journals_with_cats_metrics_wos')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking var'
        title = 'WOS - Variance of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')
        journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_scopus')

        # journals_with_cats_metrics_ranking_scopus = self.utils.load_obj('journals_with_cats_metrics_ranking_scopus')
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
            journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        plt_by = 'Percentile ranking var'
        title = 'Scopus - Variance of percentile ranking by number of categories'
        self.vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        self.vis.plt_clear()
        print('done')


    def get_outlier_journals(self,df, wos=True):
        name = 'Total_num_small_cats_wos'
        small_cats=self.get_small_categories(df, name)
        journals=set()
        for idx in range(len(small_cats)-1):
            cat=small_cats.index[idx]
            if wos:
                journals=journals.union(set(df.T[cat]['journals']['Journal title'].values))
            else:
                journals = journals.union(set(df.T[cat]['journals']))
        return journals

    def calc_mean_sd(self, df_wos, df_scopus):
        # df = self.utils.load_obj('wos_to_scopus_categories_for_group_mapping_v3')
        df_wos['count journals_in_cat'] = df_wos.apply(lambda row: pd.Series(len(row['journals'])), axis=1)
        total=df_wos['count journals_in_cat'].sum()
        mean=df_wos['count journals_in_cat'].mean()
        median=df_wos['count journals_in_cat'].median()
        std=df_wos['count journals_in_cat'].std()
        print('WOS number of journals per category - median {}, mean {}, sd {}, total {}.'.format(median,mean,std,total))
        # df = (self.utils.load_obj('scopus_to_wos_categories_for_group_mapping_v2'))
        df_scopus['count journals_in_cat'] = df_scopus.apply(lambda row: pd.Series(len(row['journals'])), axis=1)
        total = df_scopus['count journals_in_cat'].sum()
        mean = df_scopus['count journals_in_cat'].mean()
        median = df_scopus['count journals_in_cat'].median()
        std = df_scopus['count journals_in_cat'].std()
        print('Scopus number of journals per category - median {}, mean {}, sd {}, total {}.'.format(median,mean, std, total))


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

    def prep_data_for_greedy(self, wos, row):
        # self.utils.load_obj("categories_with_ranks_df_wos")
        # for wos_category, row in df.iterrows():
        if wos:
            covering_categories=row['WOS Categories']
        else:
            covering_categories=row['categories']
        cats_set=set()
        for idx, cat_string in covering_categories.items():
            cats_list=cat_string.split('|')
            cats_set.update(set(cats_list))
        if '' in cats_set:
            cats_set.remove('')
        else:
            print('cats are {}'.format(cats_set))
        return cats_set


    def run_cover_set_per_cat(self, df, coverset, wos=True, threshold=1):


        df_results = pd.DataFrame(
                columns=['Category', 'Num journals', 'Num matching cats', 'Min cover set Greedy', 'Min Cover set ILP'])
        for category, row in df.items():
            cats_set=self.prep_data_for_greedy(wos,row)
            # sco_cats_dict = row['scopus_categories']
            if len(cats_set)==0:
                print('empty')
                continue;
            cats_dict=(df.loc[cats_set]).to_dict()
            cats_dict.pop(category)
            journals = set(row['Journal title'])
            cover_set_greedy = coverset.cover_set_greedy(journals_set=journals, cats_dict=cats_dict, threshold=threshold)
            greedy_cover_set_size = len(cover_set_greedy)
            if greedy_cover_set_size>0 and threshold==1:
                const_arr = coverset.build_array_intra(row, cats_dict=cats_dict)
                ilp_model = coverset.build_model(const_arr)
                status, objective = coverset.run_model(ilp_model)
            # else:
            #     objective='0'
            #     status='NAN'
                print('Cat {}, status {}, objective {}'.format(category, status, objective))
                record = {'Category': category, 'Num journals': len(journals), 'Num matching cats': len(cats_dict),
                      'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': int(objective)}
                df_results = df_results.append(record, ignore_index=True)
            if greedy_cover_set_size>0 and threshold<1:
                record = {'Category': category, 'Num journals': len(journals), 'Num matching cats': len(cats_dict),
                          'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': 0}
                df_results = df_results.append(record, ignore_index=True)
        print(df_results)
        return df_results

    def add_cats_data(self, df_scopus_cats_with_ranks, scopus_journals_dict):
        for cat_name, cat_data in df_scopus_cats_with_ranks.items():
            cat_data['categories']=np.nan
            indexes_to_remove=[]
            for idx, journal in cat_data.iterrows():
                journal_name=journal['Journal title']
                if journal_name in scopus_journals_dict.keys():
                    journal_data=scopus_journals_dict[journal_name]
                    journal_categories=journal_data['categories']
                    cats_str = '|' + '|'.join(journal_categories) + '|'
                    cat_data.loc[
                        (cat_data['Journal title'] == journal_name), 'categories'] = cats_str
                else:
                    print('missing data for journal {}'.format(journal_name))
                    indexes_to_remove.append(idx)
            cat_data.drop(index=indexes_to_remove, inplace=True)
                # cat_data['categories']=journal_categories
        # for journal, data in scopus_journals_dict.items():
        #     for cat in data['categories']:
        return df_scopus_cats_with_ranks

    def run_cover_set(self):
        df_wos_cats_with_ranks = self.utils.load_obj("categories_with_ranks_df_wos")
        wos_coverset_df = self.run_cover_set_per_cat(df_wos_cats_with_ranks, self.cover_set, wos=True, threshold=0.95)
        self.utils.save_obj(wos_coverset_df, 'cover_set_wos_intra_0.95')
        df_scopus_cats_with_ranks = self.utils.load_obj("categories_with_ranks_df_scopus")
        scopus_coverset_df = self.run_cover_set_per_cat(df_scopus_cats_with_ranks, self.cover_set, wos=False, threshold=0.95)
        self.utils.save_obj(scopus_coverset_df, 'cover_set_scopus_intra_0.95')

    def analyse_cover_set(self):
        wos_coverset=self.utils.load_obj("cover_set_wos_intra_0.95")
        wos_coverset.rename(columns={'Min cover set Greedy': 'Min Cover set Greedy'}, inplace=True)
        scopus_coverset = self.utils.load_obj("cover_set_scopus_intra_0.95")
        scopus_coverset.rename(columns={'Min cover set Greedy': 'Min Cover set Greedy'}, inplace=True)
        scopus_coverset_max_journals=scopus_coverset['Num journals'].values.max()
        scopus_coverset_max_cover = scopus_coverset['Min Cover set ILP'].values.max()
        self.vis.plt_coverset_size(wos_coverset, scopus_coverset, label1='WOS', label2='Scopus', title1="Minimal cover size \n by number of journals", title2= "Minimal cover size \n by number of covering categories",cover_set_method='Greedy')
        print(wos_coverset)

    def prep_data_for_graph(self, dict):
        df=self.utils.load_obj("scopus_to_wos_categories_for_group_mapping")
        d = pd.DataFrame(0, index=list(dict.keys()),
                         columns=list(dict.keys()))
        graph_df=pd.DataFrame(columns=['category','intersecting category','num journals in intersect'])
        for cat_name, item in dict.items():
            d.loc[cat_name,cat_name]= len(df[cat_name]['journals'])
            for intersect_cat in item.iterrows():
                name_intersect_cat=intersect_cat[1]['category']
                ratio=intersect_cat[1]['ratio']
                num_journals = len(intersect_cat[1]['journals'])
                if (ratio > 0.1):
                    d.loc[cat_name, name_intersect_cat] = num_journals

                if not name_intersect_cat in dict.keys():
                    print('name mismatch. in cat {} can not find cat {}'.format(cat_name,name_intersect_cat))

                record={'category':cat_name, 'intersecting category': name_intersect_cat,'num journals in intersect': num_journals}
                graph_df=graph_df.append(record, ignore_index=True)
        print(graph_df.nunique())
        return graph_df, d

    def generate_graph(self,df):
        for cat,row in df.iterrows():
            df.loc[cat,cat]=0

        # df1=df[['category','intersecting category']]
        G = nx.Graph()
        G=nx.from_pandas_adjacency(df)
        # G = nx.from_pandas_edgelist(df1, 'category', 'intersecting category')
        self.vis.show_graph(G, nx)