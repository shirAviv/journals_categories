from utils import Utils
from datetime import date,datetime,timedelta
import pickle
import pandas as pd
from scipy import stats
from scipy.stats import shapiro
from visualization import Visualization
import numpy as np




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


if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    vis=Visualization()
    extractMetrics=ExtractMetrics()
    # journals_with_cats_metrics=extractMetrics.extract_wos_metrics('wos_journals_metrics.csv')
    # utils.save_obj(journals_with_cats_metrics,"wos_journals_with_metrics")

    # journals_with_cats_metrics=extractMetrics.extract_scopus_metrics('scopus_scores_2019.csv')
    # utils.save_obj(journals_with_cats_metrics, "scopus_journals_with_metrics")

    # journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus= extractMetrics.find_missing_journals(journals_with_cats_metrics_wos,journals_with_cats_metrics_scopus)
    # utils.save_obj(journals_with_cats_metrics_wos,"wos_journals_with_metrics")
    # utils.save_obj(journals_with_cats_metrics_scopus, "scopus_journals_with_metrics")

    journals_with_cats_metrics_wos = utils.load_obj("wos_journals_with_metrics")
    # vis.plt_histogram_cats(journals_with_cats_metrics_wos, title="WOS number of categories per journal distribution")
    extractMetrics.get_wos_correlations(journals_with_cats_metrics_wos)

    journals_with_cats_metrics_scopus = utils.load_obj("scopus_journals_with_metrics")
    # vis.plt_histogram_cats(journals_with_cats_metrics_scopus, title="Scopus number of categories per journal distribution")
    extractMetrics.get_scopus_correlations(journals_with_cats_metrics_scopus)

    extractMetrics.get_inter_system_correlations(journals_with_cats_metrics_wos, journals_with_cats_metrics_scopus)



