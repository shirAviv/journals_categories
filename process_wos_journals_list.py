from utils import Utils
import pandas as pd
from datetime import date,datetime,timedelta



path='D:\\shir\\study\\bibliometrics\\journals'

class ProcessWOSJournals:

    # def __init__(self):
        # self.utils=Utils(path)

    def get_wos_categories_and_journals(self, cat_file, journals_file):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        categories_dict=categories.set_index('Categories').T.to_dict('list')
        df=utils.load_csv_data_to_df(journals_file)
        for category in categories_dict:
            categories_dict[category]=dict()
            cond=df['Web of Science Categories'].str.contains(category)
            journals=df.loc[cond,:]
            categories_dict[category]['journals']=journals
        return categories_dict, df


    def get_scopus_categories_and_journals(self, cat_file, journals_file):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        # categories_dict=categories.to_dict()
        df=utils.load_csv_data_to_df(journals_file)
        return categories, df

    def match_categories_from_wos(self,wos_categories_dict, scopus_categories, scopus_journals):
        for category in wos_categories_dict:
            category_df = wos_categories_dict[category]['journals']
            count_journals=0
            if not category_df.empty:
                wos_categories_dict[category]['scopus_categories']=dict()
                for wos_index, row in category_df.iterrows():
                    journal_name=row['Journal title'].lower()
                    journal_ISSN = row['ISSN'].replace('-','')
                    journal_eISSN = row['eISSN'].replace('-','')
                    # print(journal_name)
                    cond=scopus_journals['Source Title'].str.lower()==journal_name
                    match = scopus_journals.loc[cond, :]
                    if len(match)!=1:
                        # print('not a single match, trying with ISSN')
                        cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                        match = scopus_journals.loc[cond, :]
                        if len(match) != 1:
                            # print('not a single match, trying with eISSN')
                            cond = scopus_journals['eISSN'].str.lower() == journal_eISSN
                            match = scopus_journals.loc[cond, :]
                            if len(match) == 0:
                                print('not a single match for \'{}\', giving up'.format(journal_name))
                                continue
                            if len(match) > 1:
                                print('multiple matches for \'{}\', giving up'.format(journal_name))
                                continue

                    count_journals += 1
                    scopus_subject_codes=list(match['ASJC'].str.split(';'))
                    for subject_code in scopus_subject_codes[0]:
                        if len(subject_code)==0:
                            continue
                        cond = scopus_categories['Code'] == subject_code.strip()
                        subject_name=scopus_categories.loc[cond,'Field'].iloc[0]
                        scopus_matched_categories=wos_categories_dict[category]['scopus_categories']
                        if not subject_name in wos_categories_dict[category]['scopus_categories']:
                            wos_categories_dict[category]['scopus_categories'][subject_name] = 0
                        wos_categories_dict[category]['scopus_categories'][subject_name]+=1

                print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
                print('scopus matching categories {}'.format(wos_categories_dict[category]['scopus_categories']))








if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    pwj=ProcessWOSJournals()
    wos_categories_dict, wos_df=pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_AHCI.csv')
    scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv')
    pwj.match_categories_from_wos(wos_categories_dict,scopus_categories,scopus_df)

