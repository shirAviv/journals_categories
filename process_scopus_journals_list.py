from utils import Utils
import pandas as pd
from datetime import date,datetime,timedelta
import numpy as np
from visualization import Visualization
from itertools import chain,combinations
from process_wos_journals_list import ProcessWOSJournals



path='D:\\shir\\study\\bibliometrics\\journals'

class ProcessScopusJournals:
    def get_scopus_categories_and_journals(self, cat_file, journals_file):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        # categories_dict=categories.to_dict()
        df=utils.load_csv_data_to_df(journals_file)
        return categories, df

    def remove_dash(self, issn, eIssn):
        journal_ISSN = issn.replace('-', '')
        journal_eISSN = eIssn.replace('-', '')
        return journal_ISSN, journal_eISSN

    def create_scopus_categories_dict(self,scopus_categories, scopus_journals, wos_df):
        scopus_categories_dict = dict()
        indexes_to_drop = []
        count_journals=0
        wos_df[['ISSN','eISSN']]=wos_df.apply(lambda row: pd.Series(self.remove_dash(row['ISSN'], row['eISSN'])), axis=1)
        for scopus_index, row in scopus_journals.iterrows():
            journal_name = row['Source Title'].lower()
            journal_ISSN = row['ISSN']
            journal_eISSN = row['eISSN']
            cond = wos_df['ISSN'].str.lower() == journal_ISSN
            match = wos_df.loc[cond, :]
            if len(match) != 1:
                # print('not a single match, trying with eISSN')
                if journal_ISSN.startswith('0'):
                    journal_ISSN = journal_ISSN.lstrip('0')
                else:
                    journal_ISSN = '0' + journal_ISSN
                cond = wos_df['ISSN'].str.lower() == journal_ISSN
                match = wos_df.loc[cond, :]
                if len(match) != 1:
                    cond = wos_df['eISSN'].str.lower() == journal_eISSN
                    match = wos_df.loc[cond, :]
                    if len(match) != 1:
                        cond = wos_df['Journal title'].str.lower() == journal_name
                        match = wos_df.loc[cond, :]
                        if len(match) == 0:
                            # print('not a single match for \'{}\', giving up'.format(journal_name))
                            indexes_to_drop.append(scopus_index)
                            continue
                        if len(match) > 1:
                            print('multiple matches for \'{}\', giving up'.format(journal_name))
                            indexes_to_drop.append(scopus_index)
                            continue

            # category_name=row['Field']
            # category_code=row['Code']

                count_journals += 1
                scopus_subject_codes=list(row['ASJC'].split(';'))
                    # print(scopus_subject_codes)
                for subject_code in scopus_subject_codes:
                        if len(subject_code)==0:
                            continue
                        if subject_code.strip()=='3330':
                            print('missing subject name for code 3330 in scopus')
                            continue
                        cond = scopus_categories['Code'] == subject_code.strip()
                        subject_name=scopus_categories.loc[cond,'Field'].iloc[0]
                        scopus_matched_categories=scopus_categories_dict
                        if not subject_name in scopus_categories_dict:
                            scopus_categories_dict[subject_name] = pd.DataFrame(columns=['journal name', 'ISSN','eISSN'])
                        record= {'journal name': journal_name,'ISSN':journal_ISSN, 'eISSN':journal_eISSN}
                        scopus_categories_dict[subject_name]=scopus_categories_dict[subject_name].append(record, ignore_index=True)
                # if len(indexes_to_drop)>0 :
                #     category_df.drop(index=indexes_to_drop, inplace=True)
                # sorted_scopus_categories_dict=sorted(scopus_categories_dict.items(), key=lambda x: x[1], reverse=True)
                #print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
                #print('scopus matching categories {}'.format(sorted_scopus_categories_dict))
                # wos_categories_dict[category]['scopus_categories']=scopus_categories_dict
        return scopus_categories_dict

    def match_categories_from_scopus(self, scopus_categories_dict, wos_df):
        # wos_df[['ISSN', 'eISSN']] = wos_df.apply(lambda row: pd.Series(self.remove_dash(row['ISSN'], row['eISSN'])),
        #                                          axis=1)
        scopus_full_mapping = pd.DataFrame(columns=['journals', 'wos_categories'])
        scopus_categories_dict_full=dict()
        for scopus_cat, journals_data in scopus_categories_dict.items():
            scopus_categories_dict_full[scopus_cat]=dict()
            # scopus_full_dict[scopus_cat]=dict()
            # scopus_full_dict[scopus_cat]['journals']=journals_list
            wos_categories_dict = dict()
            journals_list=set()
            for row in journals_data.iterrows():
                journal_name = row[1]['journal name'].lower()
                journal_ISSN = row[1]['ISSN']
                journal_eISSN = row[1]['eISSN']
                cond = wos_df['ISSN'].str.lower() == journal_ISSN
                match = wos_df.loc[cond, :]
                if len(match) != 1:
                    # print('not a single match, trying with eISSN')
                    if journal_ISSN.startswith('0'):
                        journal_ISSN = journal_ISSN.lstrip('0')
                    else:
                        journal_ISSN = '0' + journal_ISSN
                    cond = wos_df['ISSN'].str.lower() == journal_ISSN
                    match = wos_df.loc[cond, :]
                    if len(match) != 1:
                        cond = wos_df['eISSN'].str.lower() == journal_eISSN
                        match = wos_df.loc[cond, :]
                        if len(match) != 1:
                            cond = wos_df['Journal title'].str.lower() == journal_name
                            match = wos_df.loc[cond, :]
                            if len(match) == 0:
                                print('not a single match for \'{}\', giving up'.format(journal_name))
                                continue
                            if len(match) > 1:
                                print('multiple matches for \'{}\', giving up'.format(journal_name))
                                continue
                journals_list.add(journal_name)
                wos_cats_names = match['WOS Categories'].str.split('|').values[0]
                for wos_cat in wos_cats_names:
                    wos_cat=wos_cat.strip()
                    if not wos_cat in wos_categories_dict:
                        wos_categories_dict[wos_cat] = []
                    wos_categories_dict[wos_cat].append(journal_name)
            # record = {'journals': list(journals_list), 'wos_categories': wos_categories_dict}
            scopus_categories_dict_full[scopus_cat]['journals']=list(journals_list)
            scopus_categories_dict_full[scopus_cat]['wos_categories'] = wos_categories_dict
        scopus_full_mapping=pd.DataFrame.from_dict(scopus_categories_dict_full)

        return scopus_full_mapping

    def categories_mapping(self, scopus_to_wos_categories):
        mapping=pd.DataFrame()
        for sc_category, row in scopus_to_wos_categories.iterrows():
            wos_categories=row['wos_categories']
            num_scopus_journals=len(row['journals'])
            if num_scopus_journals==0:
                continue
            # if isinstance(scopus_categories,float):
            #     continue
            for wos_category, wos_journals in wos_categories.items():
                mapping.loc[wos_category,sc_category]=len(wos_journals)
            mapping.loc['Total', sc_category]=num_scopus_journals
            print(sc_category, num_scopus_journals)
            print(wos_categories)
        return mapping


    def extract_matches(self, scopus_wos_mapping_df):
        totals_df=scopus_wos_mapping_df.loc['Total'].copy()
        scopus_wos_mapping_df.drop(index='Total', inplace=True)
        maxValuesObj = scopus_wos_mapping_df.max()
        maxValueIndexObj = scopus_wos_mapping_df.idxmax()
        a=maxValueIndexObj.duplicated()
        df_thresholds=pd.DataFrame(columns=['Threshold','Number of Categories'])
        for step in range(0,100,5):
            threshold=step/100
            count_above_threshold=(maxValuesObj / totals_df) >= threshold
            categories_with_match=maxValueIndexObj.loc[count_above_threshold]
            count_below_threshold=(maxValuesObj / totals_df) < threshold
            categories_without_match=maxValueIndexObj.loc[count_below_threshold]
            num_categories_with_match = len(categories_with_match)
            print('for threshold {}, matched {} categories'.format(threshold, num_categories_with_match))
            row={'Threshold':threshold,'Number of Categories':num_categories_with_match}
            df_thresholds=df_thresholds.append(row, ignore_index=True)
        df_thresholds.set_index('Threshold', inplace=True)
        return df_thresholds






if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    psj=ProcessScopusJournals()
    pwj=ProcessWOSJournals()
    vis=Visualization()
    # scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv', utils)
    # wos_df = pwj.get_full_wos_df('wos-core_AHCI.csv', 'wos-core_SCIE.csv',
    #                                                                    'wos-core_SSCI.csv',  utils)
    # utils.save_obj(wos_df,'wos_df_no_dupes')
    # scopus_categories_and_journals_dict=psj.create_scopus_categories_dict(scopus_categories,scopus_df,wos_df)
    # utils.save_obj(scopus_categories_and_journals_dict,"scopus_categories_and_journals_dict")

    # wos_df=utils.load_obj('wos_df_no_dupes')
    # scopus_categories_and_journals_dict=utils.load_obj("scopus_categories_and_journals_dict")
    # scopus_full_mapping = psj.match_categories_from_scopus(scopus_categories_and_journals_dict,wos_df=wos_df)
    # print(scopus_categories_and_journals_dict)
    # utils.save_obj(scopus_full_mapping,'scopus_to_wos_categories_for_group_mapping')

    # df=utils.load_obj('scopus_to_wos_categories_for_group_mapping')
    # mapping=psj.categories_mapping(df.T)
    # utils.save_obj(mapping,'scopus_to_wos_categories_mapping')

    mapping_scop_to_wos=utils.load_obj('scopus_to_wos_categories_mapping')
    df_thresholds_scop_to_wos=psj.extract_matches(scopus_wos_mapping_df=mapping_scop_to_wos)
    mapping_wos_to_scop = utils.load_obj('wos_to_scopus_categories_mapping')
    df_thresholds_wos_to_scop = pwj.extract_matches(wos_scopus_mapping_df=mapping_wos_to_scop)
    vis.plt_match_by_threshold(df_thresholds_wos_to_scop, df_thresholds_scop_to_wos,'Categories match by threshold')

