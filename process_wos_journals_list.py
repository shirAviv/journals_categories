from utils import Utils
import pandas as pd
from datetime import date,datetime,timedelta
import numpy as np
from visualization import Visualization
from itertools import chain,combinations
from process_scopus_journals_list import ProcessScopusJournals
import pickle



class ProcessWOSJournals:


    def get_wos_categories_and_journals(self, cat_file, journals_file_SCIE,journals_file_SSCI,journals_file_AHCI,journals_file_ESCI, utils=None):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        categories_dict = categories.set_index('Categories').T.to_dict('list')
        full_df = self.get_full_wos_df(journals_file_AHCI, journals_file_SCIE,
                                                        journals_file_SSCI, journals_file_ESCI,utils=utils)
        full_df[['WOS Categories','num WOS Cats']]=full_df.apply(lambda row: pd.Series(self.remove_repeating_categories(row['WOS Categories'])), axis=1)
        # full_df['num WOS Cats']=full_df.apply(lambda row: pd.Series(self))
        for category in categories_dict:
            categories_dict[category]=dict()
            cond=full_df['WOS Categories'].str.find('|'+category+'|')!=-1
            journals=full_df.loc[cond,:]
            if not journals['Journal title'].is_unique:
                print('not unique journal in cat {}'.format(category))
            categories_dict[category]['journals']=journals
        return categories_dict, full_df

    def get_wos_categories_and_journals_ESCI(self, cat_file, journals_file_ESCI, utils=None):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        categories_dict = categories.set_index('Categories').T.to_dict('list')
        full_df = self.get_full_wos_df_ESCI(journals_file_ESCI, utils)
        full_df[['WOS Categories','num WOS Cats']]=full_df.apply(lambda row: pd.Series(self.remove_repeating_categories(row['WOS Categories'])), axis=1)
        # full_df['num WOS Cats']=full_df.apply(lambda row: pd.Series(self))
        for category in categories_dict:
            categories_dict[category]=dict()
            cond=full_df['WOS Categories'].str.find('|'+category+'|')!=-1
            journals=full_df.loc[cond,:]
            if not journals['Journal title'].is_unique:
                print('not unique journal in cat {}'.format(category))
            categories_dict[category]['journals']=journals
        return categories_dict, full_df

    def get_full_wos_df_ESCI(self, journals_file_ESCI, utils=None):
        df1 = utils.load_csv_data_to_df(journals_file_ESCI)
        full_df = df1
        full_df.sort_values(by='Journal title', kind='mergesort', inplace=True, ignore_index=True)
        dupes_df = full_df.loc[full_df.duplicated(['Journal title'], keep=False), :].copy()
        dupes_df.sort_values(by='Journal title', kind='mergesort', inplace=True)
        cond = dupes_df['Journal title'] == dupes_df["Journal title"].shift(1)
        idxs = dupes_df.index
        idxs_index = 0
        journal_name = 'unknown'
        for idx in idxs:
            if journal_name != dupes_df.loc[idx]['Journal title']:
                journal_name = dupes_df.loc[idx]['Journal title']
                occurence = 1
            else:
                occurence += 1
                if occurence == 2:
                    full_df.loc[idxs[idxs_index - 1]]['WOS Categories'] += '| ' + full_df.loc[idx]['WOS Categories']
                if occurence == 3:
                    full_df.loc[idxs[idxs_index - 2]]['WOS Categories'] += '| ' + full_df.loc[idx][
                        'WOS Categories']
                if occurence > 3 or occurence == 1:
                    print('error')
            idxs_index += 1
        full_df.drop_duplicates(['Journal title'], inplace=True, ignore_index=True)
        full_df['Journal title']=full_df['Journal title'].str.lower()
        full_df['ISSN']=full_df['ISSN'].str.lower()
        full_df['eISSN']=full_df['eISSN'].str.lower()
        full_df[['ISSN','eISSN']]=full_df.apply(lambda row: pd.Series(self.remove_leading_zeros(row['ISSN'], row['eISSN'])), axis=1)
        full_df[['ISSN','eISSN']]=full_df.apply(lambda row: pd.Series(self.remove_dash(row['ISSN'], row['eISSN'])), axis=1)

        return full_df

    def get_full_wos_df(self, journals_file_AHCI, journals_file_SCIE, journals_file_SSCI, journals_file_ESCI, utils=None):
        df1 = utils.load_csv_data_to_df(journals_file_SCIE)
        df2 = utils.load_csv_data_to_df(journals_file_SSCI)
        df3 = utils.load_csv_data_to_df(journals_file_AHCI)
        df4 = utils.load_csv_data_to_df(journals_file_ESCI)

        full_df = ((df1.append(df2)).append(df3)).append(df4)
        full_df.sort_values(by='Journal title', kind='mergesort', inplace=True, ignore_index=True)
        dupes_df = full_df.loc[full_df.duplicated(['Journal title'], keep=False), :].copy()
        dupes_df.sort_values(by='Journal title', kind='mergesort', inplace=True)
        cond = dupes_df['Journal title'] == dupes_df["Journal title"].shift(1)
        idxs = dupes_df.index
        idxs_index = 0
        journal_name = 'unknown'
        for idx in idxs:
            if journal_name != dupes_df.loc[idx]['Journal title']:
                journal_name = dupes_df.loc[idx]['Journal title']
                occurence = 1
            else:
                occurence += 1
                if occurence == 2:
                    full_df.loc[idxs[idxs_index - 1]]['WOS Categories'] += '| ' + full_df.loc[idx]['WOS Categories']
                if occurence == 3:
                    full_df.loc[idxs[idxs_index - 2]]['WOS Categories'] += '| ' + full_df.loc[idx][
                        'WOS Categories']
                if occurence > 3 or occurence == 1:
                    print('error')
            idxs_index += 1
        full_df.drop_duplicates(['Journal title'], inplace=True, ignore_index=True)
        full_df['Journal title']=full_df['Journal title'].str.lower()
        full_df['ISSN']=full_df['ISSN'].str.lower()
        full_df['eISSN']=full_df['eISSN'].str.lower()
        full_df[['ISSN','eISSN']]=full_df.apply(lambda row: pd.Series(self.remove_leading_zeros(row['ISSN'], row['eISSN'])), axis=1)
        full_df[['ISSN','eISSN']]=full_df.apply(lambda row: pd.Series(self.remove_dash(row['ISSN'], row['eISSN'])), axis=1)

        return full_df

    def remove_repeating_categories(self, cats):
        cats_list = cats.split('|')
        cats_set=set()
        for e in cats_list:
            e=e.strip()
            cats_set.add(e)
        cats_str= '|'+'|'.join(cats_set)+'|'
        num_cats=len(cats_set)
        return cats_str, num_cats

    def remove_leading_zeros(self, issn, eIssn):
        journal_ISSN = issn.lstrip('0')
        journal_eISSN = eIssn.lstrip('0')
        return journal_ISSN, journal_eISSN

    def remove_dash(self, issn, eIssn):
        journal_ISSN = issn.replace('-', '')
        journal_eISSN = eIssn.replace('-', '')
        return journal_ISSN, journal_eISSN

    def match_categories_from_wos(self,wos_categories_dict, scopus_categories, scopus_journals):
        for category in wos_categories_dict:
            category_df = wos_categories_dict[category]['journals']
            count_journals=0
            if not category_df.empty:
                print('WOS category {}'.format(category))
                scopus_categories_dict=dict()
                indexes_to_drop=[]
                for wos_index, row in category_df.iterrows():
                    journal_name=row['Journal title'].lower()
                    journal_ISSN = row['ISSN'].replace('-','')
                    journal_eISSN = row['eISSN'].replace('-','')
                    # print(journal_name)
                    cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                    match = scopus_journals.loc[cond, :]
                    if len(match) != 1:
                        # print('not a single match, trying with eISSN')
                        # if journal_ISSN.startswith('0'):
                        #     journal_ISSN=journal_ISSN.lstrip('0')
                        # else:
                        #     journal_ISSN = '0'+journal_ISSN
                        # cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                        # match = scopus_journals.loc[cond, :]
                        if len(match) != 1:
                            cond = scopus_journals['eISSN'].str.lower() == journal_eISSN
                            match = scopus_journals.loc[cond, :]
                            if len(match) != 1:
                                cond = scopus_journals['Journal title'].str.lower() == journal_name
                                match = scopus_journals.loc[cond, :]
                                if len(match) == 0:
                                    # print('not a single match for \'{}\', giving up'.format(journal_name))
                                    indexes_to_drop.append(wos_index)
                                    continue
                                if len(match) > 1:
                                    print('multiple matches for \'{}\', giving up'.format(journal_name))
                                    indexes_to_drop.append(wos_index)
                                    continue

                    count_journals += 1
                    scopus_subject_codes=list(match['ASJC'].str.split(';'))
                    # print(scopus_subject_codes)
                    for subject_code in scopus_subject_codes[0]:
                        if len(subject_code)==0:
                            continue
                        if subject_code.strip()=='3330':
                            print('missing subject name for code 3330 in scopus')
                            continue
                        cond = scopus_categories['Code'] == subject_code.strip()
                        subject_name=scopus_categories.loc[cond,'Field'].iloc[0]
                        scopus_matched_categories=scopus_categories_dict
                        if not subject_name in scopus_categories_dict:
                            scopus_categories_dict[subject_name] = 0
                        scopus_categories_dict[subject_name]+=1
                if len(indexes_to_drop)>0 :
                    category_df.drop(index=indexes_to_drop, inplace=True)
                sorted_scopus_categories_dict=sorted(scopus_categories_dict.items(), key=lambda x: x[1], reverse=True)
                #print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
                #print('scopus matching categories {}'.format(sorted_scopus_categories_dict))
                wos_categories_dict[category]['scopus_categories']=sorted_scopus_categories_dict
        wos_to_scopus_categories_df=pd.DataFrame.from_dict(wos_categories_dict)
        return wos_to_scopus_categories_df.T


    def match_categories_from_wos_2(self,wos_categories_dict, scopus_categories, scopus_journals):
       # wos_categories_dict['Acoustics']
        wos_journals_dict=dict()
        for category in wos_categories_dict:
            category_df = wos_categories_dict[category]['journals']
            count_journals=0
            if not category_df.empty:
                print('WOS category {}'.format(category))
                scopus_categories_dict=dict()
                indexes_to_drop=[]
                for wos_index, row in category_df.iterrows():
                    journal_name=row['Journal title'].lower()
                    journal_ISSN = row['ISSN'].replace('-','')
                    journal_eISSN = row['eISSN'].replace('-','')
                    if (journal_ISSN=='8846812' or journal_ISSN=='39438'):
                        print(journal_name)
                    if ( journal_eISSN=='15463222' or journal_eISSN=='19201214' or journal_eISSN=='18722040'):
                        print(journal_name)
                    cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                    match = scopus_journals.loc[cond, :]
                    if len(match) != 1:
                        if len(match)>1:
                            cond=match['Active or Inactive']=='Active'
                            match=match.loc[cond,:]
                        if len(match)!=1:
                        # print('not a single match, trying with eISSN')
                        # if journal_ISSN.startswith('0'):
                        #     journal_ISSN=journal_ISSN.lstrip('0')
                        # else:
                        #     journal_ISSN = '0'+journal_ISSN
                        # cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                        # match = scopus_journals.loc[cond, :]
                        # if len(match) != 1:
                            cond = scopus_journals['eISSN'].str.lower() == journal_eISSN
                            match = scopus_journals.loc[cond, :]
                            if len(match) != 1:
                                if len(match) > 1:
                                    cond = match['Active or Inactive'] == 'Active'
                                    match = match.loc[cond, :]
                            if len(match) != 1:
                                cond = scopus_journals['Journal title'].str.lower() == journal_name
                                match = scopus_journals.loc[cond, :]
                                if len(match) == 0:
                                    # print('not a single match for \'{}\', giving up'.format(journal_name))
                                    indexes_to_drop.append(wos_index)
                                    continue
                                if len(match) > 1:
                                    cond = match['Active or Inactive'] == 'Active'
                                    match = match.loc[cond, :]
                                    if len(match)!=1:
                                        print('multiple matches for \'{}\', giving up'.format(journal_name))
                                        indexes_to_drop.append(wos_index)
                                        continue

                    scopus_journal_active = match['Active or Inactive'].values[0]
                    if scopus_journal_active == 'Inactive':
                        continue
                    count_journals += 1
                    scopus_subject_codes=list(match['ASJC'].str.split(';'))
                    scopus_journal_name=match['Journal title']
                    if len(scopus_journal_name)==0:
                        print('error missing name in scopus for {}, ISSN {}, eISSN {}'.format(journal_name,journal_ISSN,journal_eISSN))
                        scopus_journal_name=journal_name
                    # print(scopus_subject_codes)
                    for subject_code in scopus_subject_codes[0]:
                        if len(subject_code)==0:
                            continue
                        if subject_code.strip()=='3330':
                            print('missing subject name for code 3330 in scopus')
                            continue

                        cond = scopus_categories['Code'] == subject_code.strip()
                        if len(scopus_categories.loc[cond,'Field'])==0:
                            print(journal_name)
                            print(subject_code)
                            continue
                        subject_name=scopus_categories.loc[cond,'Field'].iloc[0]
                        scopus_matched_categories=scopus_categories_dict
                        if not subject_name in scopus_categories_dict:
                            scopus_categories_dict[subject_name] = []
                        scopus_categories_dict[subject_name].append(scopus_journal_name.values[0])
                    # wos_categories_dict[category]['journals']['Journal title'][wos_index] = \
                    # wos_categories_dict[category]['journals']['Journal title'][wos_index].replace(journal_name,
                    #                                                                               scopus_journal_name.values[0])
                    wos_categories_dict[category]['journals']['Scopus Journal title'][wos_index]=scopus_journal_name.values[0]
                    if not journal_name in wos_journals_dict:
                        wos_journals_dict[journal_name]=dict()
                        wos_journals_dict[journal_name]['categories'] = set()
                        wos_journals_dict[journal_name]['scopus_journal_name'] = scopus_journal_name.values[0]

                    wos_journals_dict[journal_name]['categories'].add(category)
                if len(indexes_to_drop)>0 :
                    category_df.drop(index=indexes_to_drop, inplace=True)

                # sorted_scopus_categories_dict=sorted(scopus_categories_dict.items(), key=lambda x: x[1], reverse=True)
                #print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
                #print('scopus matching categories {}'.format(sorted_scopus_categories_dict))
                wos_categories_dict[category]['scopus_categories']=scopus_categories_dict
        wos_to_scopus_categories_df=pd.DataFrame.from_dict(wos_categories_dict)
        return wos_to_scopus_categories_df.T, wos_journals_dict

    def categories_mapping(self, wos_to_scopus_categories):
        mapping=pd.DataFrame()
        for wos_category, row in wos_to_scopus_categories.iterrows():
            scopus_categories=row['scopus_categories']
            num_wos_journals=len(row['journals'])
            if num_wos_journals==0:
                continue
            # if isinstance(scopus_categories,float):
            #     continue
            for scopus_category, num_journals in scopus_categories.items():
                mapping.loc[scopus_category,wos_category]=len(num_journals)
            mapping.loc['Total', wos_category]=num_wos_journals
            # print(wos_category, num_wos_journals)
            # print(scopus_categories)
        return mapping


    def extract_matches(self, wos_scopus_mapping_df):
        totals_df=wos_scopus_mapping_df.loc['Total'].copy()
        wos_scopus_mapping_df.drop(index='Total', inplace=True)
        maxValuesObj = wos_scopus_mapping_df.max()
        maxValueIndexObj = wos_scopus_mapping_df.idxmax()
        a=maxValueIndexObj.duplicated()
        df_thresholds=pd.DataFrame(columns=['Threshold','Percent of Categories'])
        for step in range(0,100,5):
            threshold=step/100
            count_above_threshold=(maxValuesObj / totals_df) >= threshold
            categories_with_match=maxValueIndexObj.loc[count_above_threshold]
            count_below_threshold=(maxValuesObj / totals_df) < threshold
            categories_without_match=maxValueIndexObj.loc[count_below_threshold]
            num_categories_with_match = len(categories_with_match)
            percent_categories_with_match=num_categories_with_match*100/len(totals_df)
            print('for threshold {}, matched {} categories'.format(threshold, percent_categories_with_match))
            row={'Threshold':threshold,'Percent of Categories':percent_categories_with_match}
            df_thresholds=df_thresholds.append(row, ignore_index=True)
        df_thresholds.set_index('Threshold', inplace=True)
        return df_thresholds

    def get_scopus_categories_for_all_journals(self, df):
        scopus_categories_dict = dict()
        indexes_to_drop = []
        count_journals=0
        for wos_index, row in df.iterrows():
            scopus_cats = row['scopus_categories']
            if isinstance(scopus_cats,float):
                continue
            for cat, val in scopus_cats.items():
                if not cat in scopus_categories_dict:
                    scopus_categories_dict[cat] = set()
                scopus_categories_dict[cat].update(set(val))
        # wos_to_scopus_categories_df = pd.DataFrame.from_dict(scopus_categories_dict)
        return scopus_categories_dict



    def find_same_group(self, groups_dict):
        same_groups_list=[]
        for k,v in groups_dict.items():
            groups=v['groups']['same_group']
            if len(groups)>0:
                groups.append(k)
                print('same groups {}'.format(groups))
                same_groups_list.append(groups)


    def run_groups_for_all_wos_journals(self):
        df = utils.load_obj('no_cat_wos_to_scopus_categories_for_group_mapping')
        groups_dict = pwj.find_super_groups_and_intersection_all_journals(df)
        pwj.find_sup_group(groups_dict)






    def order_categories_by_size(self, df):
        categories=df['journals']
        categories_dict=dict()
        for index, category in categories.items():
            num_journals_in_cat=len(category)
            if not num_journals_in_cat in categories_dict.keys():
                categories_dict[num_journals_in_cat]=dict()
            categories_dict[num_journals_in_cat][index]=category
        categories_dict=dict(sorted(categories_dict.items(), key=lambda item: item[0]))
        # grouped = categories.groupby(lambda x: len(categories[x]))
        # cat_by_num_journals=grouped.count()
        return categories_dict







if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils()
    pwj=ProcessWOSJournals()
    vis=Visualization()
    psj=ProcessScopusJournals()
    # wos_categories_dict, wos_df=pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv', utils)
    # scopus_categories, scopus_df=psj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv',utils)

    # wos_to_scopus_categories_df, wos_journals_dict=pwj.match_categories_from_wos_2(wos_categories_dict,scopus_categories,scopus_df)
    # utils.write_to_csv(wos_to_scopus_categories_df,'wos_to_scopus_categories_for_group_mapping.csv')
    # utils.save_obj(wos_to_scopus_categories_df,'wos_to_scopus_categories_for_group_mapping')
    # utils.save_obj(wos_journals_dict, "wos_journals_dict")


    # mapping=pwj.categories_mapping(df)
    # utils.save_obj(mapping,'wos_to_scopus_categories_mapping')
    # df1=utils.load_obj('wos_to_scopus_categories_mapping')
    # df_thresholds=pwj.extract_matches(df1)
    # vis.plt_match_by_threshold(df_thresholds,'Scopus to WOS categories match')

    # cover_set=pwj.run_cover_set_per_category()
    # vis.plt_coverset_size(cover_set[['Num journals','Cover set size']], 'test')

    # wos_df=pwj.get_full_wos_df( 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv')
    # scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv')
    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
    all_wos_journals_to_scopus_categories_dict=pwj.get_scopus_categories_for_all_journals(df1)
    utils.save_obj(all_wos_journals_to_scopus_categories_dict,'no_cat_wos_to_scopus_categories_for_group_mapping')
    # pwj.run_groups_for_all_wos_journals()

    # print(df)
