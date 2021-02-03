from utils import Utils
from visualization import Visualization
from datetime import date,datetime,timedelta
from process_wos_journals_list import ProcessWOSJournals
from process_scopus_journals_list import ProcessScopusJournals



path='D:\\shir\\study\\bibliometrics\\journals'


class RunTests:

    def create_mapping(self):
        df_s_to_c = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        # all_scopus_journals_to_wos_categories_dict = psj.get_wos_categories_for_all_journals(df_s_to_c.T)
        # utils.save_obj(all_scopus_journals_to_wos_categories_dict, 'no_cat_scopus_to_wos_categories_for_group_mapping')
        # pwj.run_groups_for_all_wos_journals()
        # mapping=psj.categories_mapping(df_s_to_c.T)
        # utils.save_obj(mapping,'scopus_to_wos_categories_mapping')
        df_c_to_s = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
        mapping = pwj.categories_mapping(df_c_to_s)
        utils.save_obj(mapping,'wos_to_scopus_categories_mapping')

    def run_thresholds(self):
        mapping_scop_to_wos = utils.load_obj('scopus_to_wos_categories_mapping')
        df_thresholds_scop_to_wos = psj.extract_matches(scopus_wos_mapping_df=mapping_scop_to_wos)
        mapping_wos_to_scop = utils.load_obj('wos_to_scopus_categories_mapping')
        df_thresholds_wos_to_scop = pwj.extract_matches(wos_scopus_mapping_df=mapping_wos_to_scop)
        vis.plt_match_by_threshold(df_thresholds_wos_to_scop, df_thresholds_scop_to_wos,
                                   'Categories match by threshold')



if __name__ == '__main__':
    start_time = datetime.now()
    print(start_time)
    utils=Utils(path=path)
    psj=ProcessScopusJournals()
    pwj=ProcessWOSJournals()
    vis=Visualization()
    runTests=RunTests()
    runTests.create_mapping()
    runTests.run_thresholds()
