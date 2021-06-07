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


path='C:\\shir\\study\\bibliometrics\\journals'

utils=Utils()
pwj=ProcessWOSJournals()
psj=ProcessScopusJournals()
cs=CoverSet(path=utils.path)
vis=Visualization()
wos_df_full = pwj.get_full_wos_df('wos-core_AHCI.csv','wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_ESCI.csv',utils)
print(len(wos_df_full))
exit(0)
# wos_categories_dict, wos_df=pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv', utils)
#scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv', utils)
# scopus_categories_and_journals_dict=psj.create_scopus_categories_dict(scopus_categories,scopus_df,wos_df)
# utils.save_obj(scopus_categories_and_journals_dict,"scopus_categories_and_journals_dict")

wos_journals_dict = utils.load_obj("wos_journals_dict")
scopus_journals_dict = utils.load_obj("scopus_journals_dict")


scopus_categories_and_journals_dict=utils.load_obj("scopus_categories_and_journals_dict")

def fix_esci_in_wos():
    wos_categories_dict_esci, wos_df_esci=pwj.get_wos_categories_and_journals_ESCI('wos_categories.csv','wos-core_ESCI.csv',utils=utils)
    scopus_categories, scopus_df=psj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv',utils)
    wos_to_scopus_categories_df_esci, wos_journals_dict_esci=pwj.match_categories_from_wos_2(wos_categories_dict_esci,scopus_categories,scopus_df)
    utils.write_to_csv(wos_to_scopus_categories_df_esci,'wos_to_scopus_categories_for_group_mapping_ESCI.csv')
    utils.save_obj(wos_to_scopus_categories_df_esci,'wos_to_scopus_categories_for_group_mapping_esci')
    utils.save_obj(wos_journals_dict_esci, "wos_journals_dict_esci")
    #
    print(len(wos_to_scopus_categories_df_esci))

    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v1')
    df1_esci=utils.load_obj('wos_to_scopus_categories_for_group_mapping_esci')
    df1_journals=df1['journals']
    df1_esci_journals=df1_esci['journals']
    for cat,item in df1_journals.iteritems():
        esci_cat=df1_esci_journals[cat]
        indexes_to_remove=set()
        for idx,row in esci_cat.iterrows():
            journal_name = row['Journal title'].lower()
            journal_ISSN = row['ISSN'].replace('-', '')
            journal_eISSN = row['eISSN'].replace('-', '')
            cond = item['ISSN'].str.lower() == journal_ISSN
            match = item.loc[cond, :]
            if len(match) != 0 and len(journal_ISSN)>0:
                indexes_to_remove.add(idx)
            else:
                cond = item['eISSN'].str.lower() == journal_eISSN
                match = item.loc[cond, :]
                if len(match) != 0 and len(journal_eISSN)>0:
                    indexes_to_remove.add(idx)
                else:
                    cond = item['Journal title'].str.lower() == journal_name
                    match = item.loc[cond, :]
                    if len(match) != 0:
                        indexes_to_remove.add(idx)

        if len(indexes_to_remove)>0:
            esci_cat.drop(index=indexes_to_remove, inplace=True)
        df1_journals[cat]=df1_journals[cat].append(esci_cat)
    print('done journals')
    df1_scopus_cats=df1['scopus_categories']
    df1_esci_scopus_cats=df1_esci['scopus_categories']
    for cat,item in df1_scopus_cats.iteritems():
        esci_cat=df1_esci_scopus_cats[cat]
        indexes_to_remove=set()
        if isinstance(esci_cat,float):
            continue
        for scopus_cat_name,scopus_journals_esci in esci_cat.items():
            scopus_journals_esci_len = len(scopus_journals_esci)
            if scopus_cat_name in item.keys():
                scopus_journals=set(item[scopus_cat_name])
                scopus_journals_len=len(scopus_journals)
                scopus_journals.update(set(scopus_journals_esci))

                if len(scopus_journals)!=scopus_journals_len+scopus_journals_esci_len:
                    print('repeating journals exist {} in scopus cat {}, in wos cat {}'.format(scopus_journals,scopus_cat_name,cat))
                item[scopus_cat_name]=list(scopus_journals)
            else:
                item[scopus_cat_name]=list(scopus_journals_esci)
    print('done journals')
    utils.save_obj(df1,'wos_to_scopus_categories_for_group_mapping_v2')
    count_new_journals=set()
    for cat, row in df1['journals'].iteritems():
        count_new_journals.update(set(row['Journal title']))

    print(len(count_new_journals))

def fix_esci_in_scopus():
        # scopus_categories, scopus_df=psj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv', utils)
        # wos_df_esci = pwj.get_full_wos_df('wos-core_ESCI.csv',utils)
        # wos_df_esci = utils.load_csv_data_to_df('wos-core_ESCI.csv')

        # wos_df_esci.drop_duplicates(['Journal title'], inplace=True, ignore_index=True)
        # wos_df_esci['Journal title'] = wos_df_esci['Journal title'].str.lower()
        # wos_df_esci['ISSN'] = wos_df_esci['ISSN'].str.lower()
        # wos_df_esci['eISSN'] = wos_df_esci['eISSN'].str.lower()
        # wos_df_esci[['ISSN', 'eISSN']] = wos_df_esci.apply(
        #     lambda row: pd.Series(psj.remove_leading_zeros(row['ISSN'], row['eISSN'])), axis=1)
        # wos_df_esci[['ISSN', 'eISSN']] = wos_df_esci.apply(lambda row: pd.Series(psj.remove_dash(row['ISSN'], row['eISSN'])),
        #                                            axis=1)
        # utils.save_obj(wos_df_esci,'wos_df_no_dupes_esci')
        # wos_df=utils.load_obj('wos_df_no_dupes_esci')

        # scopus_categories_and_journals_dict_esci, scopus_journals_dict_esci=psj.create_scopus_categories_dict(scopus_categories,scopus_df,wos_df_esci)
        # utils.save_obj(scopus_journals_dict_esci, "scopus_journals_dict_esci")
        # utils.save_obj(scopus_categories_and_journals_dict_esci,"scopus_categories_and_journals_dict_esci")

        # wos_df=utils.load_obj('wos_df_no_dupes_esci')
        # scopus_categories_and_journals_dict=utils.load_obj("scopus_categories_and_journals_dict_esci")
        # scopus_full_mapping_esci = psj.match_categories_from_scopus(scopus_categories_and_journals_dict_esci,wos_df=wos_df_esci)
        # print(scopus_categories_and_journals_dict)
        # utils.save_obj(scopus_full_mapping_esci,'scopus_to_wos_categories_for_group_mapping_esci')
        df2_esci=utils.load_obj('scopus_to_wos_categories_for_group_mapping_esci')
        df2_esci=df2_esci.T
        df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        df2=df2.T
        df2_journals = df2['journals']
        df2_esci_journals = df2_esci['journals']
        for cat, item in df2_journals.iteritems():
            journals_in_scopus=set(item)
            len_journals_in_scopus=len(journals_in_scopus)
            if cat in df2_esci_journals.keys():
                esci_cat = set(df2_esci_journals[cat])
                len_journals_in_esci=len(esci_cat)
                journals_in_scopus.update(esci_cat)
                if len(journals_in_scopus)!=len_journals_in_scopus+len_journals_in_esci:
                    print('mismatch in len for cat {} journals in scopus {}, journals in esci {} total {}'.format(cat,len_journals_in_scopus, len_journals_in_esci, len(journals_in_scopus)))
                df2_journals[cat] = list(journals_in_scopus)
        print('done journals')
        df2_wos_cats = df2['wos_categories']
        df2_esci_scopus_cats = df2_esci['wos_categories']
        for cat, item in df2_wos_cats.iteritems():
            if cat in df2_esci_scopus_cats.keys():
                esci_cat = df2_esci_scopus_cats[cat]
                indexes_to_remove = set()
                if isinstance(esci_cat, float):
                    continue
                for wos_cat_name, scopus_journals_esci in esci_cat.items():
                    scopus_journals_esci_len = len(scopus_journals_esci)
                    if wos_cat_name in item.keys():
                        scopus_journals = set(item[wos_cat_name])
                        scopus_journals_len = len(scopus_journals)
                        scopus_journals.update(set(scopus_journals_esci))

                        if len(scopus_journals) != scopus_journals_len + scopus_journals_esci_len:
                            print('repeating journals exist {} in scopus cat {}, in wos cat {}'.format(scopus_journals,
                                                                                                       wos_cat_name,
                                                                                                       cat))
                        item[wos_cat_name] = list(scopus_journals)
                    else:
                        item[wos_cat_name] = list(scopus_journals_esci)
        print('done cats')
        utils.save_obj(df2, 'scopus_to_wos_categories_for_group_mapping_v2')
        count_new_journals = set()
        for cat, row in df2['journals'].iteritems():
            count_new_journals.update(set(row))
        print(len(count_new_journals))


def remove_missing_sc_journals_from_wos():
    df1=utils.load_obj('wos_to_scopus_categories_for_group_mapping_v2')
    count_new_journals = set()
    for cat, row in df1['journals'].iteritems():
        count_missing_sc_journal_name=set()
        for idx, journal in row.iterrows():
            if len(journal['Scopus Journal title'])==0:
                count_missing_sc_journal_name.add(idx)
        if len(count_missing_sc_journal_name)>0:
            row.drop(index=count_missing_sc_journal_name, inplace=True)

            print('missing journal names {}'.format(count_missing_sc_journal_name))
        count_new_journals.update(set(row['Journal title']))
    print(len(count_new_journals))
    utils.save_obj(df1, 'wos_to_scopus_categories_for_group_mapping_v3')

    df2=utils.load_obj('scopus_to_wos_categories_for_group_mapping_v2')
    count_new_journals = set()
    for cat, row in df2['journals'].iteritems():
        count_new_journals.update(set(row))
    print(len(count_new_journals))


def fix_wos_journals_dict():
    df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping_v3')
    journals_dict = utils.load_obj("wos_journals_dict")
    journals_dict_esci = utils.load_obj("wos_journals_dict_esci")
    for k,v in journals_dict_esci.items():
        if not k in journals_dict.keys():
            journals_dict[k]=v
        else:
            print(k)
            journals_dict[k]['categories'].update(v['categories'])
    utils.save_obj(journals_dict,'wos_journals_dict_v3')
    print(len(journals_dict))

def fix_scopus_journals_dict():
    journals_dict = utils.load_obj("scopus_journals_dict")
    journals_dict_esci = utils.load_obj("scopus_journals_dict_esci")
    for k,v in journals_dict_esci.items():
        if not k in journals_dict.keys():
            journals_dict[k]=v
        else:
            print(k)
            journals_dict[k]['categories'].update(v['categories'])
    utils.save_obj(journals_dict,'scopus_journals_dict_v2')
    print(len(journals_dict))


fix_scopus_journals_dict()
exit(0)


# df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
# pwj.split_categories_by_size(df1,0.9)


# all_wos_journals_to_scopus_categories_dict = pwj.get_scopus_categories_for_all_journals(df1)
jl1=df1['scopus_categories']
js1=set()
count=0
tmp_dict=dict()
for k,v in jl1.items():
    if isinstance(v,float):
        continue
    # print(v.keys())
    cat_name=k.replace(',',';')
    tmp_dict[cat_name]=set()
    for k2, v2 in v.items():
        # if k2=='Biomedical Engineering':
        #     print(v2)
        for i in v2:
            count+=1
            # print(i)
            js1.update(list(i))
            tmp_dict[cat_name].update(list(i))
js1=sorted(list(js1))
file_path = os.path.join(utils.path, 'wos_cats_journals.csv')
# with open(file_path, 'w', encoding="utf8") as csvfile:
#     for key in tmp_dict.keys():
#         csvfile.write("%s,%s\n"%(key,tmp_dict[key]))

# utils.write_to_csv(pd.DataFrame.from_records(tmp_dict),"wos_cats_journals")
# print(js1)
dict2 = utils.load_obj('scopus_categories_and_journals_dict')
# jl2=df2.T['journals']
js2=set()
count=0
tmp_dict=dict()
for k,v in dict2.items():
    count+=len(v)
    jn=list(v['journal name'])
    cat_name=k.replace(',',';')
    tmp_dict[cat_name]=set()
    if 'biomedical materials' in jn:
        print(k)
        print(jn)
    # if name=='biomedical materials':
    #     print(name)
    js2.update(list(v['journal name']))
    tmp_dict[cat_name].update(jn)
js2=sorted(list(js2))
file_path = os.path.join(utils.path, 'scopus_cats_journals.csv')
# with open(file_path, 'w', encoding="utf8") as csvfile:
#     for key in tmp_dict.keys():
#         csvfile.write("%s,%s\n"%(key,tmp_dict[key]))

missing_journals=set()
for jo in js2:
    if not jo.lower() in js1:
        print('not found {}'.format(jo.lower()))
        missing_journals.update(jo.lower())
missing_journals=sorted(list(missing_journals))
# cs.run_cover_set_no_cat_scopus(utils)
print(len(js1))
print(len(js2))

df=utils.load_obj("wos_df_no_dupes")
print(len(df))



