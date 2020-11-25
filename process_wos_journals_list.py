from utils import Utils
import pandas as pd
from datetime import date,datetime,timedelta
import numpy as np
from visualization import Visualization
from itertools import chain,combinations



path='D:\\shir\\study\\bibliometrics\\journals'

class ProcessWOSJournals:

    # def __init__(self):
        # self.utils=Utils(path)

    def get_wos_categories_and_journals(self, cat_file, journals_file_SCIE,journals_file_SSCI,journals_file_AHCI, utils=None):
        categories = utils.load_csv_data_to_df(cat_file)
        print(categories)
        categories_dict = categories.set_index('Categories').T.to_dict('list')
        full_df = self.get_full_wos_df(journals_file_AHCI, journals_file_SCIE,
                                                        journals_file_SSCI)
        full_df[['WOS Categories','num WOS Cats']]=full_df.apply(lambda row: pd.Series(self.remove_repeating_categories(row['WOS Categories'])), axis=1)
        # full_df['num WOS Cats']=full_df.apply(lambda row: pd.Series(self))
        for category in categories_dict:
            categories_dict[category]=dict()
            cond=full_df['WOS Categories'].str.contains(category)
            journals=full_df.loc[cond,:]
            if not journals['Journal title'].is_unique:
                print('not unique journal in cat {}'.format(category))
            categories_dict[category]['journals']=journals
        return categories_dict, full_df

    def get_full_wos_df(self, journals_file_AHCI, journals_file_SCIE, journals_file_SSCI, utils=None):
        df1 = utils.load_csv_data_to_df(journals_file_SCIE)
        df2 = utils.load_csv_data_to_df(journals_file_SSCI)
        df3 = utils.load_csv_data_to_df(journals_file_AHCI)
        full_df = (df1.append(df2)).append(df3)
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
        return full_df

    def remove_repeating_categories(self, cats):
        cats_set = set(cats.split('|'))
        cats_str= ' | '.join(cats_set)
        num_cats=len(cats_set)
        return cats_str, num_cats

    def get_scopus_categories_and_journals(self, cat_file, journals_file, utils=None):
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
                        if journal_ISSN.startswith('0'):
                            journal_ISSN=journal_ISSN.lstrip('0')
                        else:
                            journal_ISSN = '0'+journal_ISSN
                        cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                        match = scopus_journals.loc[cond, :]
                        if len(match) != 1:
                            cond = scopus_journals['eISSN'].str.lower() == journal_eISSN
                            match = scopus_journals.loc[cond, :]
                            if len(match) != 1:
                                cond = scopus_journals['Source Title'].str.lower() == journal_name
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
                        if journal_ISSN.startswith('0'):
                            journal_ISSN=journal_ISSN.lstrip('0')
                        else:
                            journal_ISSN = '0'+journal_ISSN
                        cond = scopus_journals['ISSN'].str.lower() == journal_ISSN
                        match = scopus_journals.loc[cond, :]
                        if len(match) != 1:
                            cond = scopus_journals['eISSN'].str.lower() == journal_eISSN
                            match = scopus_journals.loc[cond, :]
                            if len(match) != 1:
                                cond = scopus_journals['Source Title'].str.lower() == journal_name
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
                            scopus_categories_dict[subject_name] = []
                        scopus_categories_dict[subject_name].append(journal_name)
                if len(indexes_to_drop)>0 :
                    category_df.drop(index=indexes_to_drop, inplace=True)
                # sorted_scopus_categories_dict=sorted(scopus_categories_dict.items(), key=lambda x: x[1], reverse=True)
                #print('WOS category {}, number of journals checked {} out of {}'.format(category,count_journals, len(category_df)))
                #print('scopus matching categories {}'.format(sorted_scopus_categories_dict))
                wos_categories_dict[category]['scopus_categories']=scopus_categories_dict
        wos_to_scopus_categories_df=pd.DataFrame.from_dict(wos_categories_dict)
        return wos_to_scopus_categories_df.T

    def categories_mapping(self, wos_to_scopus_categories):
        mapping=pd.DataFrame()
        for wos_category, row in wos_to_scopus_categories.iterrows():
            scopus_categories=row['scopus_categories']
            num_wos_journals=len(row['journals'])
            if num_wos_journals==0:
                continue
            # if isinstance(scopus_categories,float):
            #     continue
            for scopus_category in scopus_categories:
                mapping.loc[scopus_category[0],wos_category]=scopus_category[1]
            mapping.loc['Total', wos_category]=num_wos_journals
            print(wos_category, num_wos_journals)
            print(scopus_categories)
        return mapping


    def extract_matches(self, wos_scopus_mapping_df):
        totals_df=wos_scopus_mapping_df.loc['Total'].copy()
        wos_scopus_mapping_df.drop(index='Total', inplace=True)
        maxValuesObj = wos_scopus_mapping_df.max()
        maxValueIndexObj = wos_scopus_mapping_df.idxmax()
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

    def find_super_groups_and_intersection_all_journals(self,sco_cats_dict):
        sorted_sco_cats = sorted(sco_cats_dict, key=lambda k: len(sco_cats_dict[k]))
        for i in range(len(sorted_sco_cats)):
            sco_cat = sorted_sco_cats[i]
            sco_cat_dict = dict()
            sco_cat_dict['sup_group'] = []
            sco_cat_dict['intersect_group'] = []
            sco_cat_dict['same_group'] = []
            journals = sco_cats_dict[sorted_sco_cats[i]]

            for j in range(i + 1, len(sorted_sco_cats)):
                intersect = False
                sup_group = True
                compared_journals = sco_cats_dict[sorted_sco_cats[j]]
                for journal in journals:
                    if journal in compared_journals:
                        intersect = True
                    else:
                        sup_group = False
                if intersect == True and sup_group == True:
                    if len(journals) == len(compared_journals):
                        sco_cat_dict['same_group'].append(sorted_sco_cats[j])
                    else:
                        sco_cat_dict['sup_group'].append(sorted_sco_cats[j])
                else:
                    if intersect == True:
                        sco_cat_dict['intersect_group'].append(sorted_sco_cats[j])
            sco_cats_dict[sco_cat] = dict()
            sco_cats_dict[sco_cat]['journals'] = journals
            sco_cats_dict[sco_cat]['groups'] = sco_cat_dict
        return sco_cats_dict

    def find_same_group(self, groups_dict):
        same_groups_list=[]
        for k,v in groups_dict.items():
            groups=v['groups']['same_group']
            if len(groups)>0:
                groups.append(k)
                print('same groups {}'.format(groups))
                same_groups_list.append(groups)

    def find_sup_group(self, groups_dict):
        same_groups_list=[]
        for k,v in groups_dict.items():
            groups=v['groups']['sup_group']
            if len(groups)>0:
                sup_groups=groups
                print('cat {} has sup cats {}'.format(k,sup_groups))
                same_groups_list.append(sup_groups)


    def find_super_groups_and_intersection_per_wos_cat(self,wos_cats_df):
        groups_dict=dict()
        for wos_category, row in wos_cats_df.iterrows():
            sco_cats_dict=row['scopus_categories']
            if isinstance(sco_cats_dict,float):
                continue
            sorted_sco_cats=sorted(sco_cats_dict, key=lambda k: len(sco_cats_dict[k]))
            for i in range(len(sorted_sco_cats)):
                sco_cat=sorted_sco_cats[i]
                sco_cat_dict=dict()
                sco_cat_dict['sup_group']=[]
                sco_cat_dict['intersect_group'] = []
                sco_cat_dict['same_group'] = []
                journals=sco_cats_dict[sorted_sco_cats[i]]

                for j in range(i+1,len(sorted_sco_cats)):
                    intersect = False
                    sup_group = True
                    compared_journals=sco_cats_dict[sorted_sco_cats[j]]
                    for journal in journals:
                        if journal in compared_journals:
                            intersect=True
                        else:
                            sup_group=False
                    if intersect==True and sup_group==True:
                        if len(journals)==len(compared_journals):
                            sco_cat_dict['same_group'].append(sorted_sco_cats[j])
                        else:
                            sco_cat_dict['sup_group'].append(sorted_sco_cats[j])
                    else:
                        if intersect==True:
                            sco_cat_dict['intersect_group'].append(sorted_sco_cats[j])
                sco_cats_dict[sco_cat]=dict()
                sco_cats_dict[sco_cat]['journals']=journals
                sco_cats_dict[sco_cat]['groups']=sco_cat_dict
            print(wos_category)
            groups_dict[wos_category]=sco_cats_dict
            # for key, val in sco_cats_dict:
        return groups_dict

    def run_groups_per_wos_cat(self):
        df = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
        groups_dict = pwj.find_super_groups_and_intersection_per_wos_cat(df)
        print('\n\n\n')
        for k, v in groups_dict.items():
            print('for cat {}'.format(k))
            pwj.find_sup_group(v)
        # sco_cats_dict=groups_dict['Allergy']
        # sorted_sco_cats = sorted(sco_cats_dict, key=lambda k: len(sco_cats_dict[k]))
        # for sc_cat in sorted_sco_cats:
        #     val=sco_cats_dict[sc_cat]
        #     print('\n cat {}, num journals {}'.format(sc_cat, len(val['journals'])))
        #     groups=val['groups']
        #     sup_group=groups['sup_group']
        #     intersect_group = groups['intersect_group']
        #     same_group=groups['same_group']
        #     if (len(same_group)>0):
        #         print('same group {}'.format(same_group))
        #     if (len(sup_group)>0):
        #         print('sup group {}'.format(sup_group))
        #     if (len(intersect_group)>0):
        #         print('intersect group {}'.format(intersect_group))


    def run_groups_for_all_wos_journals(self):
        df = utils.load_obj('no_cat_wos_to_scopus_categories_for_group_mapping')
        groups_dict = pwj.find_super_groups_and_intersection_all_journals(df)
        pwj.find_sup_group(groups_dict)

    def cover_set_greedy(self,journals, cats_dict ):
        sorted_cats = sorted(cats_dict, key=lambda k: len(cats_dict[k]))
        universe=journals
        universe['Journal title']=universe.apply(lambda row: pd.Series(row['Journal title'].lower()), axis=1)
        cover=set()
        while not len(universe)==0:
            current_cat_name = sorted_cats.pop()
            current_journals = cats_dict[current_cat_name]
            # current_cat=cats_dict.pop()
            current_universe = universe[~universe['Journal title'].isin(current_journals)]
            if len(current_universe)<len(universe):
                cover.add(current_cat_name)
                universe=current_universe
        return cover

    def cover_set_brute_force(self, journals, cats_dict):
        universe=journals
        cat_names=list(cats_dict.keys())
        cats_power_group=chain.from_iterable(combinations(cat_names, r) for r in range(3,len(cat_names)+1))
        print(cats_power_group)
        for subset in cats_power_group:
            current_universe=universe
            current_cover_set_size=len(subset)
            # print(current_cover_set_size)
            for current_set in subset:
                current_journals=cats_dict[current_set]
                current_universe = current_universe[~current_universe['Journal title'].isin(current_journals)]
            if len(current_universe)==0:
                cover=subset
                print(cover)
                break
        return cover



    def run_cover_set_per_category(self):
        df = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
        cover_set_df=pd.DataFrame(columns=['Category','Num journals','Cover set size'])
        for wos_category, row in df.iterrows():
            sco_cats_dict = row['scopus_categories']
            if isinstance(sco_cats_dict, float):
                print('skipping cat {}'.format(wos_category))
                continue
            journals=row['journals']
            print('number cats {}, num journals {}'.format(len(sco_cats_dict),len(journals)))
            cover_set=self.cover_set_greedy(journals=journals, cats_dict=sco_cats_dict)
            cover_set_size=len(cover_set)
            print('cover set size for greeedy alg {}'.format(cover_set_size))
            if cover_set_size<=200:
                record = {'Category':wos_category,'Num journals':len(journals), 'Cover set size':cover_set_size}
                cover_set_df=cover_set_df.append(record, ignore_index=True)
                print('{} greedy size {}'.format(wos_category, cover_set_size))
            else:
                cover_set_2=self.cover_set_brute_force(journals=journals,cats_dict=sco_cats_dict)
                cover_set_2_size = len(cover_set_2)
                record = {'Category':wos_category,'Num journals':len(journals), 'Cover set size':cover_set_2_size}
                cover_set_df=cover_set_df.append(record, ignore_index=True)
                if cover_set_2_size >cover_set_size:
                    print('Error for {} greedy size {}, brute force {}'.format(wos_category, cover_set_size, cover_set_2_size))
                else:
                    if cover_set_2_size==cover_set_size:
                        print('no change in size for {} greedy size {}, brute force {}'.format(wos_category, cover_set_size, cover_set_2_size))
                    else:
                        print('found smaller cover set for {} greedy size {}, brute force {}'.format(wos_category,cover_set_size, cover_set_2_size))
        return cover_set_df





if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    pwj=ProcessWOSJournals()
    vis=Visualization()
    # wos_categories_dict, wos_df=pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv')
    # scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv')
    # wos_to_scopus_categories_df=pwj.match_categories_from_wos_2(wos_categories_dict,scopus_categories,scopus_df)
    # utils.write_to_csv(wos_to_scopus_categories_df,'wos_to_scopus_categories_for_group_mapping.csv')
    # utils.save_obj(wos_to_scopus_categories_df,'wos_to_scopus_categories_for_group_mapping')
    # mapping=pwj.categories_mapping(df)
    # utils.save_obj(mapping,'wos_to_scopus_categories_mapping')
    df1=utils.load_obj('wos_to_scopus_categories_mapping')
    df_thresholds=pwj.extract_matches(df1)
    # vis.plt_match_by_threshold(df_thresholds,'Scopus to WOS categories match')

    cover_set=pwj.run_cover_set_per_category()
    vis.plt_coverset_size(cover_set[['Num journals','Cover set size']], 'test')

    # wos_df=pwj.get_full_wos_df( 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv')
    # scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv')
    # all_wos_journals_to_scopus_categories_dict=pwj.get_scopus_categories_for_all_journals(df)
    # utils.save_obj(all_wos_journals_to_scopus_categories_dict,'no_cat_wos_to_scopus_categories_for_group_mapping')
    # pwj.run_groups_for_all_wos_journals()

    # print(df)
