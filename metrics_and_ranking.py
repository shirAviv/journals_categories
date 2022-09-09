from utils import Utils
from process_wos_journals_list import ProcessWOSJournals
from cover_set import CoverSet
from process_scopus_journals_list import ProcessScopusJournals
from visualization import Visualization
import pandas as pd
import csv
import os
import pickle
import numpy as np
from extract_metrics_intra import ExtractMetricsIntra



def create_scopus_categories_dict(scopus_categories, scopus_journals_df):
    scopus_categories_dict = dict()
    indexes_to_drop = []
    count_journals = 0
    scopus_journals_dict = dict()
    scopus_journals_df['categories']=''

    # wos_df[['ISSN','eISSN']]=wos_df.apply(lambda row: pd.Series(self.remove_dash(row['ISSN'], row['eISSN'])), axis=1)
    for scopus_index, row in scopus_journals_df.iterrows():
        categories=[]
        journal_name = row['Journal title'].lower()
        journal_ISSN = row['ISSN']
        journal_eISSN = row['eISSN']
        sourcerecord_id = row['Sourcerecord ID']
        journal_active = row['Active or Inactive']
        if journal_active == 'Inactive':
            continue
        count_journals += 1
        scopus_subject_codes = list(row['ASJC'].split(';'))
        citeScore = row['2019 CiteScore']
        # print(scopus_subject_codes)
        for subject_code in scopus_subject_codes:
            if len(subject_code) == 0:
                continue
            if subject_code.strip() == '3330':
                print('missing subject name for code 3330 in scopus')
                continue
            cond = scopus_categories['Code'] == subject_code.strip()
            if len(scopus_categories.loc[cond, 'Field']) == 0:
                print(journal_name)
                print(subject_code)
                continue
            subject_name = scopus_categories.loc[cond, 'Field'].iloc[0]
            categories.append(subject_name)
            scopus_matched_categories = scopus_categories_dict
            if not subject_name in scopus_categories_dict:
                scopus_categories_dict[subject_name] = pd.DataFrame(columns=['journal name', 'ISSN', 'eISSN'])
            record = {'journal name': journal_name, 'ISSN': journal_ISSN, 'eISSN': journal_eISSN}
            scopus_categories_dict[subject_name] = scopus_categories_dict[subject_name].append(record,
                                                                                               ignore_index=True)
            # if len(indexes_to_drop)>0 :
            #     category_df.drop(index=indexes_to_drop, inplace=True)
            # sorted_scopus_categories_dict=sorted(scopus_categories_dict.items(), key=lambda x: x[1], reverse=True)
            # print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
            # print('scopus matching categories {}'.format(sorted_scopus_categories_dict))
            # wos_categories_dict[category]['scopus_categories']=scopus_categories_dict
            if not journal_name in scopus_journals_dict:
                scopus_journals_dict[journal_name] = dict()
                scopus_journals_dict[journal_name]['categories'] = set()
                scopus_journals_dict[journal_name]['CiteScore'] = citeScore
                scopus_journals_dict[journal_name]['ISSN'] = journal_ISSN
                scopus_journals_dict[journal_name]['eISSN'] = journal_eISSN
                scopus_journals_dict[journal_name]['sourcerecord_id'] = sourcerecord_id

            scopus_journals_dict[journal_name]['categories'].add(subject_name)
        scopus_journals_df.loc[scopus_journals_df['Journal title'] == journal_name, 'categories'] = '|'.join(categories)
    print('count journals {}'.format(count_journals))
    return scopus_categories_dict, scopus_journals_dict

def create_wos_journals_dict(wos_journals_df):
    wos_journals_dict=dict()
    for wos_index, row in wos_journals_df.iterrows():
        journal_name = row['Journal title'].lower()
        num_categories=row['num WOS Cats']
        if not journal_name in wos_journals_dict:
            wos_journals_dict[journal_name] = dict()
            wos_journals_dict[journal_name]['num categories'] = num_categories
    return wos_journals_dict


def extract_wos_metrics(metrics_file_wos,journals_dict):
        metrics_df = utils.load_csv_data_to_df(metrics_file_wos)
        metrics_df['Full Journal Title'] = metrics_df.apply(lambda row: pd.Series(row['Full Journal Title'].lower()),
                                                            axis=1)
        count_missing = 0
        journals_with_cats_metrics=pd.DataFrame(columns={'Journal name', 'num categories', 'Total Cites', 'JIF', '5 year JIF', 'Eigenfactor', 'Norm Eigenfactor'})
        for journal_name, item in journals_dict.items():
            num_categories=item['num categories']
            metrics=metrics_df[metrics_df['Full Journal Title']==journal_name].copy()
            if len(metrics)==0:
                print("no metrics for {}".format(journal_name))
                count_missing+=1
                record={'Journal name':journal_name, 'num categories':num_categories, 'Total Cites':np.nan, 'JIF':np.nan, '5 year JIF':np.nan, 'Eigenfactor':np.nan, 'Norm Eigenfactor':np.nan}

            else:
                # JIF=metrics['Journal Impact Factor'].values[0]
                # if not np.nan(JIF):
                record={'Journal name':journal_name, 'num categories':num_categories, 'Total Cites':int(metrics['Total Cites'].values[0].replace(',','')), 'JIF':metrics['Journal Impact Factor'].values[0], '5 year JIF':metrics['5-Year Impact Factor'].values[0], 'Eigenfactor':float(metrics['Eigenfactor Score'].values[0]), 'Norm Eigenfactor':float(metrics['Normalized Eigenfactor'].values[0])}
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
        journals_with_cats_metrics=journals_with_cats_metrics[np.isnan(journals_with_cats_metrics['JIF'])==False]
        utils.save_obj(journals_with_cats_metrics,"wos_all_journals_with_metrics")

        return journals_with_cats_metrics


def create_clusters_by_categories_wos(remove_outliers=False, outlier_journals=None, metric='JIF'):
        journals_with_cats_metrics = utils.load_obj("wos_all_journals_with_metrics")
        journals_with_cats_metrics_no_outlier_JIF=journals_with_cats_metrics[journals_with_cats_metrics.JIF<1000]
        if remove_outliers:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_JIF[~journals_with_cats_metrics_no_outlier_JIF['Journal name'].isin(outlier_journals)]
        else:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_JIF
        groups_by_num_categories=journals_with_cats_metrics_for_analysis.groupby('num categories')
        plt_by=metric
        title = metric+' statistics by number of categories'
        vis.plt_group_data_with_box_plot(journals_with_cats_metrics_for_analysis, x='num categories', plt_by=plt_by, title=title)
        vis.plt_clear()
        title = 'Highest '+metric+' by number of categories'
        vis.plt_groups_max(groups_by_num_categories, plt_by=plt_by, title=title)
        vis.plt_clear()


def create_clusters_by_categories_scopus(remove_outliers=False, outlier_journals=None,metric='SJR'):
        # self.vis.plt_group_date_with_box_plot(groups_by_num_categories, plt_by=plt_by, title=title)
        journals_with_cats_metrics = utils.load_obj("scopus_all_journals_with_metrics")
        # journals_with_cats_metrics_no_outlier_citescore = journals_with_cats_metrics[
        #     journals_with_cats_metrics.CiteScore <= 200]
        # if remove_outliers:
        #     journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_citescore[~journals_with_cats_metrics_no_outlier_citescore['Journal name'].isin(outlier_journals)]
        # else:
        #     journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_citescore
        # groups_by_num_categories = journals_with_cats_metrics_for_analysis.groupby('num categories')

        journals_with_cats_metrics_no_outlier_sjr = journals_with_cats_metrics[
            journals_with_cats_metrics['num categories'] <= 10]
        # journals_with_cats_metrics_no_outlier_sjr = journals_with_cats_metrics[
        # journals_with_cats_metrics.SJR <= 700]
        if remove_outliers:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_sjr[~journals_with_cats_metrics_no_outlier_sjr['Journal name'].isin(outlier_journals)]
        else:
            journals_with_cats_metrics_for_analysis=journals_with_cats_metrics_no_outlier_sjr
        groups_by_num_categories = journals_with_cats_metrics_for_analysis.groupby('num categories')
        plt_by = metric
        title = metric+' statistics by number of categories'
        vis.plt_group_data_with_box_plot(journals_with_cats_metrics_for_analysis, x='num categories',
                                         plt_by=plt_by, title=title)
        vis.plt_clear()
        # self.vis.plt_groups_data(groups_by_num_categories, plt_by=plt_by, title=title)
        title = 'Highest '+ metric+' by number of categories'
        vis.plt_groups_max(groups_by_num_categories, plt_by=plt_by, title=title)
        vis.plt_clear()

def create_wos_dict_with_metrics(wos_df,journals_with_cats_metrics, wos_categories_dict):
    print(journals_with_cats_metrics)
    journals_names=list(journals_with_cats_metrics['Journal name'].values)
    for name in journals_names:
        JIF=journals_with_cats_metrics.loc[journals_with_cats_metrics['Journal name']==name,'JIF'].values[0]
        Eigen=journals_with_cats_metrics.loc[journals_with_cats_metrics['Journal name']==name,'Eigenfactor'].values[0]
        NormEigen=journals_with_cats_metrics.loc[journals_with_cats_metrics['Journal name']==name,'Norm Eigenfactor'].values[0]

        wos_cats_names = wos_df.loc[wos_df['Journal title'] == name, 'WOS Categories'].str.split('|').values[0]
        for wos_cat in wos_cats_names:
            wos_cat = wos_cat.strip()
            if len(wos_cat)>0:
                wos_categories_dict[wos_cat]['journals'].loc[wos_categories_dict[wos_cat]['journals']['Journal title']==name,'JIF']=JIF
                wos_categories_dict[wos_cat]['journals'].loc[wos_categories_dict[wos_cat]['journals']['Journal title']==name,'Eigenfactor']=Eigen
                wos_categories_dict[wos_cat]['journals'].loc[wos_categories_dict[wos_cat]['journals']['Journal title']==name,'NormEigenfactor']=NormEigen

    rm_cats=[]
    for cat,item in wos_categories_dict.items():
        journals=item['journals']
        if not 'JIF' in journals.columns:
            print(cat)
            rm_cats.append(cat)
            continue
        to_remove = journals.loc[np.isnan(journals['JIF'])]
        if len(to_remove)>0:
            index_to_remove=to_remove.index
            journals.drop(index=index_to_remove, inplace=True)
        journals.sort_values(by=['JIF'], inplace=True, ascending=False)
        # journals.reset_index(inplace=True)
        journals['Percentile Rank JIF'] = journals.JIF.rank(pct=True)
        journals.sort_values(by=['Eigenfactor'], inplace=True, ascending=False)
        # journals.reset_index(inplace=True)
        journals['Percentile Rank Eigenfactor'] = journals.Eigenfactor.rank(pct=True)
        journals.sort_values(by=['NormEigenfactor'], inplace=True, ascending=False)
        # journals.reset_index(inplace=True)
        journals['Percentile Rank NormEigenfactor'] = journals.NormEigenfactor.rank(pct=True)
    for rm_cat in rm_cats:
        wos_categories_dict.pop(rm_cat)
    utils.save_obj(wos_categories_dict,'wos_categories_with_rank_dict')

    # wos_df_only_metric=wos_df.loc[wos_df['Journal title'].isin(journals_names)]
    # utils.save_obj(wos_df_only_metric,"full_wos_df")

def create_scopus_dict_with_metrics(scopus_df,journals_with_cats_metrics, scopus_categories_dict):
    journals_names=list(journals_with_cats_metrics['Journal name'].values)
    # scopus_df_only_metric=scopus_df.loc[scopus_df['Journal title'].isin(journals_names)]
    # utils.save_obj(scopus_df_only_metric,"full_scopus_df")
    print(journals_with_cats_metrics)
    for name in journals_names:
        SJR=journals_with_cats_metrics.loc[journals_with_cats_metrics['Journal name']==name,'SJR'].values[0]
        CiteScore=journals_with_cats_metrics.loc[journals_with_cats_metrics['Journal name']==name,'CiteScore'].values[0]

        scopus_cats_names = scopus_df.loc[scopus_df['Journal title'] == name, 'categories'].str.split('|').values[0]
        for cat in scopus_cats_names:
            cat = cat.strip()
            if len(cat)>0:
                scopus_categories_dict[cat].loc[scopus_categories_dict[cat]['journal name']==name,'SJR']=SJR
                scopus_categories_dict[cat].loc[scopus_categories_dict[cat]['journal name'] == name, 'CiteScore'] = CiteScore
    rm_cats=[]
    for cat,journals in scopus_categories_dict.items():

        if not 'SJR' in journals.columns:
            print(cat+'SJR')
            rm_cats.append(cat)
            continue
        if not 'CiteScore' in journals.columns:
            print(cat+'CiteScore')
        to_remove = journals.loc[np.isnan(journals['SJR'])]
        if len(to_remove)>0:
            index_to_remove=to_remove.index
            journals.drop(index=index_to_remove, inplace=True)
        journals.sort_values(by=['SJR'], inplace=True, ascending=False)
        # journals.reset_index(inplace=True)
        journals['Percentile Rank SJR'] = journals.SJR.rank(pct=True)
        journals.sort_values(by=['CiteScore'], inplace=True, ascending=False)
        # journals.reset_index(inplace=True)
        journals['Percentile Rank CiteScore'] = journals.CiteScore.rank(pct=True)
    for rm_cat in rm_cats:
        scopus_categories_dict.pop(rm_cat)
    utils.save_obj(scopus_categories_dict,'scopus_categories_with_rank_dict_CiteScore')




def get_categories_ranking_mismatch_wos(wos_df,journals_with_cats_metrics, categories_with_ranks_dict,metric='JIF'):
    journals_with_cats_metrics['Percentile ranking Max Min Range '+metric] = np.nan
    journals_with_cats_metrics['Percentile ranking ASOS '+metric] = np.nan
    journals_with_cats_metrics['Percentile ranking var '+metric] = np.nan
    journals_with_cats_metrics['Percentile ranking mean '+metric] = np.nan
    journals_with_cats_metrics['Top Percentile ranking '+metric] = np.nan
    journals_with_cats_metrics['Lowest Percentile ranking '+metric] = np.nan


    for idx,row in journals_with_cats_metrics.iterrows():
        rank_metric = []
        journal_name=row['Journal name']
        wos_cats_names = wos_df.loc[wos_df['Journal title'] == journal_name, 'WOS Categories'].str.split('|').values[0]
        for wos_cat in wos_cats_names:
            wos_cat = wos_cat.strip()
            if len(wos_cat)==0:
                continue
            journals_in_cat = categories_with_ranks_dict[wos_cat]['journals']
            metrics = journals_in_cat[journals_in_cat['Journal title'] == journal_name].copy()
            if len(metrics) == 0:
                rank_metric.append(np.nan)
            else:
                rank_metric.append(metrics['Percentile Rank '+metric].values[0] * 100)
        rank_metric = np.array(rank_metric)
        if len(rank_metric)==0:
            continue
        max_rank = rank_metric.max()
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Top Percentile ranking '+metric] = max_rank
        min_rank = rank_metric.min()
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Lowest Percentile ranking '+metric] = min_rank
        if row['num categories'] == 1:
            continue
        max_min_range = rank_metric.ptp()

        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics[
                 'Journal name'] == journal_name), 'Percentile ranking Max Min Range '+metric] = max_min_range
        avg_sum_of_squares = np.linalg.norm(rank_metric) / np.size(rank_metric)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics[
                 'Journal name'] == journal_name), 'Percentile ranking ASOS '+metric] = avg_sum_of_squares
        var_metric = np.var(rank_metric)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking var '+metric] = var_metric
        mean_metric = np.mean(rank_metric)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking mean '+metric] = mean_metric
    utils.save_obj(journals_with_cats_metrics,'wos_all_journals_with_metrics_and_rank_data '+metric)

def get_categories_ranking_mismatch_scopus(scopus_df,journals_with_cats_metrics, categories_with_ranks_dict, metric='SJR'):
    journals_with_cats_metrics['Percentile ranking Max Min Range ' + metric] = np.nan
    journals_with_cats_metrics['Percentile ranking ASOS ' + metric] = np.nan
    journals_with_cats_metrics['Percentile ranking var ' + metric] = np.nan
    journals_with_cats_metrics['Percentile ranking mean ' + metric] = np.nan
    journals_with_cats_metrics['Top Percentile ranking ' + metric] = np.nan
    journals_with_cats_metrics['Lowest Percentile ranking ' + metric] = np.nan


    for idx,row in journals_with_cats_metrics.iterrows():
        rank = []
        journal_name=row['Journal name']
        cats_names = scopus_df.loc[scopus_df['Journal title'] == journal_name, 'categories'].str.split('|').values[0]
        for cat in cats_names:
            cat = cat.strip()
            if len(cat)==0:
                continue
            journals_in_cat = categories_with_ranks_dict[cat]
            metrics = journals_in_cat[journals_in_cat['journal name'] == journal_name].copy()
            if len(metrics) == 0:
                rank.append(np.nan)
            else:
                rank.append(metrics['Percentile Rank '+metric].values[0] * 100)
        rank = np.array(rank)
        if len(rank)==0:
            continue
        max_rank = rank.max()
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Top Percentile ranking '+metric] = max_rank
        min_rank = rank.min()
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Lowest Percentile ranking '+metric] = min_rank
        if row['num categories'] == 1:
            continue

        max_min_range = rank.ptp()

        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics[
                 'Journal name'] == journal_name), 'Percentile ranking Max Min Range '+metric] = max_min_range
        avg_sum_of_squares = np.linalg.norm(rank) / np.size(rank)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics[
                 'Journal name'] == journal_name), 'Percentile ranking ASOS '+metric] = avg_sum_of_squares
        var_metric = np.var(rank)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking var '+metric] = var_metric
        mean_metric = np.mean(rank)
        journals_with_cats_metrics.loc[
            (journals_with_cats_metrics['Journal name'] == journal_name), 'Percentile ranking mean '+metric] = mean_metric

    utils.save_obj(journals_with_cats_metrics,'scopus_all_journals_with_metrics_and_rank_data_'+metric)

def plot_min_max_by_cats(metric='JIF'):
        journals_with_cats_metrics_ranking_wos = utils.load_obj('wos_all_journals_with_metrics_and_rank_data_'+metric)
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking Max Min Range '+metric
        title = 'WOS - MM of percentile ranking by number of categories'
        vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        vis.plt_clear()
        print('done')
        # journals_with_cats_metrics_ranking_scopus = utils.load_obj('scopus_all_journals_with_metrics_and_rank_data_'+metric)
        # journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
        #     journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        # journals_with_multiple_categories = journals_with_multiple_categories.loc[journals_with_multiple_categories['num categories']<11]
        # plt_by = 'Percentile ranking Max Min Range '+metric
        # title = 'Scopus - MM of percentile ranking by number of categories'
        # vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
        #                                       plt_by=plt_by, title=title)
        # vis.plt_clear()
        # print('done')


def plot_avg_sum_of_squares_by_cats(metric='JIF'):
        journals_with_cats_metrics_ranking_wos = utils.load_obj('wos_all_journals_with_metrics_and_rank_data_'+metric)
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking ASOS '+metric
        title = 'WOS - ASOS of percentile ranking by number of categories'
        vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        vis.plt_clear()
        print('done')
        # journals_with_cats_metrics_ranking_scopus = utils.load_obj('scopus_all_journals_with_metrics_and_rank_data_'+metric)
        # journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
        #     journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        # journals_with_multiple_categories = journals_with_multiple_categories.loc[journals_with_multiple_categories['num categories']<11]
        # plt_by = 'Percentile ranking ASOS '+metric
        # title = 'Scopus - ASOS of percentile ranking by number of categories'
        # vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
        #                                       plt_by=plt_by, title=title)
        # vis.plt_clear()
        # print('done')


def plot_mean_variance_by_cats(metric='JIF'):
        journals_with_cats_metrics_ranking_wos = utils.load_obj('wos_all_journals_with_metrics_and_rank_data_'+metric)
        journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
            journals_with_cats_metrics_ranking_wos['num categories'] > 1]
        plt_by = 'Percentile ranking var '+metric
        title = 'WOS - Variance of percentile ranking by number of categories'
        vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                              plt_by=plt_by, title=title)
        vis.plt_clear()
        print('done')
        journals_with_cats_metrics_ranking_scopus = utils.load_obj('journals_with_cats_metrics_scopus')

        # journals_with_cats_metrics_ranking_scopus = utils.load_obj('scopus_all_journals_with_metrics_and_rank_data_'+metric)
        # journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
        #     journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
        # journals_with_multiple_categories = journals_with_multiple_categories.loc[journals_with_multiple_categories['num categories']<11]
        # plt_by = 'Percentile ranking var '+metric
        # title = 'Scopus - Variance of percentile ranking by number of categories'
        # vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
        #                                       plt_by=plt_by, title=title)
        # vis.plt_clear()
        # print('done')


def plot_mean_std_by_cats(metric='JIF'):
    journals_with_cats_metrics_ranking_wos = utils.load_obj('wos_all_journals_with_metrics_and_rank_data_' + metric)
    journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
        journals_with_cats_metrics_ranking_wos['num categories'] > 1]
    plt_by = 'Percentile ranking std '+metric
    title = 'WOS - STD of percentile ranking by number of categories'
    vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                          plt_by=plt_by, title=title)
    vis.plt_clear()
    print('done')
    # journals_with_cats_metrics_ranking_scopus = utils.load_obj('journals_with_cats_metrics_scopus')

    # journals_with_cats_metrics_ranking_scopus = utils.load_obj('scopus_all_journals_with_metrics_and_rank_data_'+metric)
    # journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
    #     journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
    # journals_with_multiple_categories = journals_with_multiple_categories.loc[
    #     journals_with_multiple_categories['num categories'] < 11]
    # plt_by = 'Percentile ranking std '+metric
    # title = 'Scopus - STD of percentile ranking by number of categories'
    # vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
    #                                  plt_by=plt_by, title=title)
    # vis.plt_clear()
    # print('done')

def plot_mean_by_cats(metric='JIF'):
    journals_with_cats_metrics_ranking_wos = utils.load_obj('wos_all_journals_with_metrics_and_rank_data_' + metric)
    journals_with_multiple_categories = journals_with_cats_metrics_ranking_wos.loc[
        journals_with_cats_metrics_ranking_wos['num categories'] > 1]
    plt_by = 'Percentile ranking mean '+metric
    title = 'WOS - mean of percentile ranking by number of categories'
    vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                          plt_by=plt_by, title=title)
    vis.plt_clear()
    print('done')
    # journals_with_cats_metrics_ranking_scopus = utils.load_obj('journals_with_cats_metrics_scopus')

    journals_with_cats_metrics_ranking_scopus = utils.load_obj('scopus_all_journals_with_metrics_and_rank_data_CiteScore')
    journals_with_multiple_categories = journals_with_cats_metrics_ranking_scopus.loc[
        journals_with_cats_metrics_ranking_scopus['num categories'] > 1]
    journals_with_multiple_categories = journals_with_multiple_categories.loc[
        journals_with_multiple_categories['num categories'] < 11]
    plt_by = 'Percentile ranking mean '+metric
    title = 'Scopus - mean of percentile ranking by number of categories'
    vis.plt_group_data_with_box_plot(journals_with_multiple_categories, x='num categories',
                                     plt_by=plt_by, title=title)
    vis.plt_clear()
    print('done')



def plot_top_ranking(wos_df,scopus_df):
    wos_df.loc[wos_df['num categories'] > 1, 'Lowest Percentile ranking'].sort_values().reset_index(drop=True).plot()
    wos_df.loc[wos_df['num categories'] > 1, 'Top Percentile ranking'].sort_values().reset_index(drop=True).plot()

    scopus_df.loc[scopus_df['num categories'] > 1, 'Lowest Percentile ranking'].sort_values().reset_index(
        drop=True).plot()
    scopus_df.loc[scopus_df['num categories'] > 1, 'Top Percentile ranking'].sort_values().reset_index(drop=True).plot()

if __name__ == '__main__':
    utils = Utils()

    pwj = ProcessWOSJournals()
    psj = ProcessScopusJournals()
    vis = Visualization()
    extractMetricsIntra = ExtractMetricsIntra()
    extractMetricsIntra.utils = utils
    extractMetricsIntra.vis = vis

    # wos_df_full = pwj.get_full_wos_df('wos-core_AHCI.csv', 'wos-core_SCIE.csv', 'wos-core_SSCI.csv',
    #                                   'wos-core_ESCI.csv', utils)
    # wos_categories_dict, wos_df = pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_SCIE.csv',
    #                                                                   'wos-core_SSCI.csv', 'wos-core_AHCI.csv',
    #                                                                   'wos-core_ESCI.csv', utils)
    # utils.save_obj(wos_categories_dict,'wos_categories_dictionary')
    # print(len(wos_df))
    # utils.save_obj(wos_df,"full_wos_df")
    # wos_journals_dict=create_wos_journals_dict(wos_df)
    # print(len(wos_journals_dict.keys()))
    # journals_with_cats_metrics=extract_wos_metrics('wos_journals_metrics.csv',wos_journals_dict)
    # print(journals_with_cats_metrics.keys())
    # exit(0)
    # scopus_categories, scopus_df = psj.get_scopus_categories_and_journals('scopus_categories_full.csv',
    #                                                                       'scopus_full_2020.csv', utils)
    # scopus_categories_dict, scopus_journals_dict = create_scopus_categories_dict(scopus_categories, scopus_df)
    # print(len(scopus_df))
    # utils.save_obj(scopus_categories_dict,'scopus_categories_dictionary')
    # utils.save_obj(scopus_df, "full_scopus_df")
    # utils.save_obj(scopus_journals_dict,'scopus_journals_dict')
    # metrics_df = utils.load_csv_data_to_df('scopus_scores_2019.csv', delimiter=';')
    # metrics_df['Title'] = metrics_df.apply(lambda row: pd.Series(row['Title'].lower()), axis=1)

    # journals_with_cats_metrics=extractMetricsIntra.extract_scopus_metrics(metrics_df,scopus_journals_dict)
    # utils.save_obj(journals_with_cats_metrics, "scopus_all_journals_with_metrics")
    #
    # extractMetricsIntra.plt_histograms_num_cats()
    # create_clusters_by_categories_wos(metric='Norm Eigenfactor')
    create_clusters_by_categories_scopus(metric='SJR')

    wos_df=utils.load_obj("full_wos_df")
    journals_with_cats_metrics = utils.load_obj("wos_all_journals_with_metrics")
    # wos_categories_dict= utils.load_obj("wos_categories_dictionary")

    # create_wos_dict_with_metrics(wos_df,journals_with_cats_metrics,wos_categories_dict)
    wos_categories_with_ranks_dict = utils.load_obj('wos_categories_with_rank_dict')
    # get_categories_ranking_mismatch_wos(wos_df, journals_with_cats_metrics,wos_categories_with_ranks_dict,'JIF')
    # wos_all_journals_with_metrics_and_rank_data=utils.load_obj('wos_all_journals_with_metrics_and_rank_data_NormEigenfactor')
    # wos_all_journals_with_metrics_and_rank_data['Percentile ranking std']=wos_all_journals_with_metrics_and_rank_data.apply(lambda row: pd.Series(np.sqrt(row['Percentile ranking var'])), axis=1)
    # utils.save_obj(wos_all_journals_with_metrics_and_rank_data,'wos_all_journals_with_metrics_and_rank_data')
    # plot_min_max_by_cats(metric='JIF')
    # plot_avg_sum_of_squares_by_cats(metric='JIF')
    # plot_mean_variance_by_cats(metric='JIF')
    # plot_mean_std_by_cats()

    scopus_df = utils.load_obj("full_scopus_df")
    journals_with_metrics = utils.load_obj("scopus_all_journals_with_metrics")
    scopus_categories_dict = utils.load_obj("scopus_categories_dictionary")

    # create_scopus_dict_with_metrics(scopus_df,journals_with_metrics,scopus_categories_dict)
    scopus_categories_with_ranks_dict = utils.load_obj('scopus_categories_with_rank_dict')
    # get_categories_ranking_mismatch_scopus(scopus_df, journals_with_metrics,scopus_categories_with_ranks_dict, metric='SJR')

    # scopus_all_journals_with_metrics_and_rank_data=utils.load_obj('scopus_all_journals_with_metrics_and_rank_data')
    # scopus_all_journals_with_metrics_and_rank_data['Percentile ranking std']=scopus_all_journals_with_metrics_and_rank_data.apply(lambda row: pd.Series(np.sqrt(row['Percentile ranking var'])), axis=1)
    # utils.save_obj(scopus_all_journals_with_metrics_and_rank_data,'scopus_all_journals_with_metrics_and_rank_data')

    # plot_min_max_by_cats(metric='SJR')
    # plot_avg_sum_of_squares_by_cats(metric='SJR')
    # plot_mean_variance_by_cats(metric='SJR')
    # plot_mean_std_by_cats(metric='CiteScore')
    # plot_mean_by_cats()
    # extractMetricsIntra.analyse_categories_ranking_mismatch()
    # plot_top_ranking(wos_all_journals_with_metrics_and_rank_data,scopus_all_journals_with_metrics_and_rank_data)
