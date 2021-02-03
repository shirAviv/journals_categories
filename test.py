from utils import Utils
from process_wos_journals_list import ProcessWOSJournals
from cover_set import CoverSet
from process_scopus_journals_list import ProcessScopusJournals
import pandas as pd
import csv
import os
import pickle


path='D:\\shir\\study\\bibliometrics\\journals'

utils=Utils(path=path)
pwj=ProcessWOSJournals()
psj=ProcessScopusJournals()
cs=CoverSet()

# wos_categories_dict, wos_df=pwj.get_wos_categories_and_journals('wos_categories.csv', 'wos-core_SCIE.csv', 'wos-core_SSCI.csv', 'wos-core_AHCI.csv', utils)
#scopus_categories, scopus_df=pwj.get_scopus_categories_and_journals('scopus_categories_full.csv', 'scopus_full_2020.csv', utils)
# scopus_categories_and_journals_dict=psj.create_scopus_categories_dict(scopus_categories,scopus_df,wos_df)
# utils.save_obj(scopus_categories_and_journals_dict,"scopus_categories_and_journals_dict")

scopus_categories_and_journals_dict=utils.load_obj("scopus_categories_and_journals_dict")
df1 = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
df2 = utils.load_obj('scopus_to_wos_categories_for_group_mapping')



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
file_path = os.path.join(path, 'wos_cats_journals.csv')
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
file_path = os.path.join(path, 'scopus_cats_journals.csv')
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



