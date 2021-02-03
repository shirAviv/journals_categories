import pulp
from collections import defaultdict
from itertools import product, combinations
import numpy as np
import pandas as pd
from utils import Utils
from datetime import date,datetime,timedelta
from visualization import Visualization




path='D:\\shir\\study\\bibliometrics\\journals'

class CoverSet:

    def cover_set_greedy(self,journals, cats_dict ):
        sorted_cats = sorted(cats_dict, key=lambda k: len(cats_dict[k]))
        universe=set(journals)
        # universe['Journal title']=universe.apply(lambda row: pd.Series(row['Journal title'].lower()), axis=1)
        cover=set()
        while not len(universe)==0:
            current_cat_name = sorted_cats.pop()
            current_journals = cats_dict[current_cat_name]
            current_journals=set(list(map(lambda x: x.values[0], current_journals)))
            # current_cat=cats_dict.pop()
            current_universe = universe-current_journals
            if len(current_universe)<len(universe):
                cover.add(current_cat_name)
                universe=current_universe
        return cover

    def calc_cover_set_ILP_pulp(self):
        model=pulp.LpProblem("minimal_set_cover", pulp.LpMinimize)
        nJournals=6
        nCats=3

        x1 = pulp.LpVariable('x1', lowBound=0, cat='Continuous')
        x2 = pulp.LpVariable('x2', lowBound=0, cat='Continuous')
        x3 = pulp.LpVariable('x3', lowBound=0, cat='Continuous')
        x4 = pulp.LpVariable('x4', lowBound=0, cat='Continuous')
        x5 = pulp.LpVariable('x5', lowBound=0, cat='Continuous')
        x6 = pulp.LpVariable('x6', lowBound=0, cat='Continuous')
        x7 = pulp.LpVariable('x7', lowBound=0, cat='Continuous')
        x8 = pulp.LpVariable('x8', lowBound=0, cat='Continuous')
        x9 = pulp.LpVariable('x9', lowBound=0, cat='Continuous')
        x10 = pulp.LpVariable('x10', lowBound=0, cat='Continuous')
        x11 = pulp.LpVariable('x11', lowBound=0, cat='Continuous')
        x12 = pulp.LpVariable('x12', lowBound=0, cat='Continuous')
        x13 = pulp.LpVariable('x13', lowBound=0, cat='Continuous')
        x14 = pulp.LpVariable('x14', lowBound=0, cat='Continuous')
        x15 = pulp.LpVariable('x15', lowBound=0, cat='Continuous')
        x16 = pulp.LpVariable('x16', lowBound=0, cat='Continuous')
        x17 = pulp.LpVariable('x17', lowBound=0, cat='Continuous')
        x18 = pulp.LpVariable('x18', lowBound=0, cat='Continuous')
        x19 = pulp.LpVariable('x19', lowBound=0, cat='Continuous')
        x20 = pulp.LpVariable('x20', lowBound=0, cat='Continuous')
        x21 = pulp.LpVariable('x21', lowBound=0, cat='Continuous')
        x22 = pulp.LpVariable('x22', lowBound=0, cat='Continuous')
        x23 = pulp.LpVariable('x23', lowBound=0, cat='Continuous')
        x24 = pulp.LpVariable('x24', lowBound=0, cat='Continuous')
        x25 = pulp.LpVariable('x25', lowBound=0, cat='Continuous')

        obj_func=x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15+x16+x17+x18+x19+x20+x21+x22+x23+x24+x25,"Z"
        model+=obj_func

        constraint1=x1+x2+x3>=1
        model += constraint1
        constraint2=x2+x3+x4+x5>=1
        model += constraint2
        constraint3=x6+x7+x8+x9>=1
        model += constraint3
        constraint4=x6+x10+x11>=1
        model += constraint4
        constraint5=x3+x4>=1
        model += constraint5
        constraint6=x3+x12+x13>=1
        model += constraint6
        constraint7=x3+x14>=1
        model += constraint7
        constraint8=x6+x15>=1
        model += constraint8
        constraint9=x3+x6+x10>=1
        model += constraint9
        constraint10=x3+x6+x15+x16+x17+x18>=1
        model += constraint10
        constraint11=x1+x3>=1
        model += constraint11
        constraint12=x1+x2+x3+x19>=1
        model += constraint12
        constraint13=x2>=1
        model += constraint13
        constraint14=x3+x20+x21>=1
        model += constraint14
        constraint15=x3+x13+x22+x23>=1
        model += constraint15
        constraint16=x6+x15+x24+x25>=1
        model += constraint16
        print(model)

        model.solve()
        print(pulp.LpStatus[model.status])

        for variable in model.variables():
            print("{} = {}".format(variable.name, variable.varValue))
        print(pulp.value(model.objective))
        # variable_names = [str(i) + str(j) for j in range(1, nJournals + 1) for i in range(1, nCats + 1)]
        # variable_names.sort()
        # print("Variable Indices:", variable_names)

    def build_array(self, df_cat=0.0, journals=None, cats_dict=None, from_scopus=False):
        if not isinstance(df_cat,float):
            if not from_scopus:
                cats_dict = df_cat['scopus_categories']
            else:
                cats_dict = df_cat['wos_categories']
            if isinstance(cats_dict, float):
                return
            if from_scopus:
                journals = pd.DataFrame(columns=['Journal title'])
                journals['Journal title'] = df_cat['journals'].copy()
            else:
                journals = df_cat['journals'].copy()
        journals.reset_index(drop=True, inplace=True)
        num_journals=len(journals)
        num_cats=len(cats_dict)
        arr=np.zeros([num_journals, num_cats])
        cats_names=list(cats_dict.keys())
        for j, cat_name in enumerate(cats_names):
            val=cats_dict[cat_name]
            for journal_name in val:
                i=journals[journals['Journal title'] == journal_name].index[0]
                arr[i,j]=1
        return arr


    def build_constraints(self, constraints_array, model):

        num_vars=constraints_array.shape[1]
        num_journals=constraints_array.shape[0]
        # obj_func=' + '.join(self.variable_name(seq) for seq in range(num_vars))

        vars_list = [pulp.LpVariable(self.variable_name(seq), lowBound=0, cat='Continuous') for seq in range(num_vars)]
        obj_func = pulp.LpAffineExpression([(vars_list[i],1) for i in range(len(vars_list))]),'Z'
        model+=obj_func

        for i in range(num_journals):
            hits=np.where(constraints_array[i] == 1)
            # constraint=' + '.join(self.variable_name(seq) for seq in hits[0])+' >= '+str(1)
            e = pulp.LpAffineExpression([(vars_list[seq], 1) for seq in hits[0]])
            constraint=pulp.LpConstraint(e,sense=pulp.const.LpConstraintGE,rhs=1)
            model+=constraint

        return

    # def build_vars(self, array, model):
    #     num_vars = array.shape[1]
    #     vars_list=[]
    #     for seq in range(num_vars):
    #         var=pulp.LpVariable(self.variable_name(seq), lowBound=0, cat='Continuous')
    #         vars_list[seq]=var
    #     return vars_list


    def build_model(self, constraints_array):


        model=pulp.LpProblem("minimal_set_cover", pulp.LpMinimize)
        self.build_constraints(constraints_array, model)

        # print(model)
        return(model)

    def run_model(self,model):
        model.solve()
        print(pulp.LpStatus[model.status])
        # for variable in model.variables():
        #     print("{} = {}".format(variable.name, variable.varValue))
        print(pulp.value(model.objective))
        return pulp.LpStatus[model.status],pulp.value(model.objective)



    def all_fill(self,source, num):
        output_len = (len(source) + num)
        for where in combinations(range(output_len), len(source)):
            poss = ([[0, 1]] * output_len)
            for (w, s) in zip(where, source):
                poss[w] = [s]
            for tup in product(*poss):
                (yield tup)

    def variable_name(self,num):
        # if not isinstance(num, int):
        #     num=num[0]
        return ('x' + ''.join(str(num)))

    def run_cover_set_per_category_wos(self):
        df = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
        df_results = pd.DataFrame(
            columns=['Category', 'Num journals', 'Num matching cats', 'Min cover set Greedy', 'Min Cover set ILP'])
        for wos_category, row in df.iterrows():
            sco_cats_dict = row['scopus_categories']
            if isinstance(sco_cats_dict, float):
                print('skipping cat {}'.format(wos_category))
                continue
            journals = list(row['journals']['Journal title'])
            cover_set_greedy = cs.cover_set_greedy(journals=journals, cats_dict=sco_cats_dict)
            greedy_cover_set_size = len(cover_set_greedy)
            # if wos_category=='Ergonomics':
            const_arr = cs.build_array(row)
            ilp_model = cs.build_model(const_arr)
            status, objective = cs.run_model(ilp_model)
            print('Cat {}, status {}, objective {}'.format(wos_category, status, objective))
            record = {'Category': wos_category, 'Num journals': len(journals), 'Num matching cats': len(sco_cats_dict),
                      'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': int(objective)}
            df_results=df_results.append(record, ignore_index=True)
        return df_results

    def run_cover_set_no_cat_wos(self, utils=None):
        sco_cats_dict = utils.load_obj('no_cat_wos_to_scopus_categories_for_group_mapping')
        journals_set=set()
        for k,v in sco_cats_dict.items():
            journals_set.update(v)
        df_results = pd.DataFrame(
            columns=['Category', 'Num journals', 'Num matching cats', 'Min cover set Greedy', 'Min Cover set ILP'])
        journals_df=pd.DataFrame(columns=['Journal title'])
        journals_df['Journal title']=list(journals_set)
        start_time = datetime.now()
        cover_set_greedy = cs.cover_set_greedy(journals=journals_df, cats_dict=sco_cats_dict)
        end_time = datetime.now()
        print(end_time - start_time)
        greedy_cover_set_size = len(cover_set_greedy)
        const_arr = cs.build_array(journals=journals_df, cats_dict= sco_cats_dict)
        ilp_model = cs.build_model(const_arr)
        status, objective = cs.run_model(ilp_model)
        print('All journals, all categories, status {}, objective {}'.format(status, objective))
        record = {'Category': 'All', 'Num journals': len(journals_set), 'Num matching cats': len(sco_cats_dict),
                  'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': int(objective)}
        return record


    def run_cover_set_per_category_scopus(self):
        df = utils.load_obj('scopus_to_wos_categories_for_group_mapping')
        df=df.T
        df_results = pd.DataFrame(
            columns=['Category', 'Num journals', 'Num matching cats', 'Min cover set Greedy', 'Min Cover set ILP'])
        for scopus_category, row in df.iterrows():
            wos_cats_dict = row['wos_categories']
            if isinstance(wos_cats_dict, float):
                print('skipping cat {}'.format(scopus_category))
                continue
            journals=pd.DataFrame(columns=['Journal title'])
            journals['Journal title'] = row['journals']
            cover_set_greedy = cs.cover_set_greedy(journals=journals, cats_dict=wos_cats_dict)
            greedy_cover_set_size = len(cover_set_greedy)
            # if scopus_category=='Ergonomics':
            const_arr = cs.build_array(row, from_scopus=True)
            ilp_model = cs.build_model(const_arr)
            status, objective = cs.run_model(ilp_model)
            print('Cat {}, status {}, objective {}'.format(scopus_category, status, objective))
            # objective=1
            record = {'Category': scopus_category, 'Num journals': len(journals), 'Num matching cats': len(wos_cats_dict),
                      'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': int(objective)}
            df_results=df_results.append(record, ignore_index=True)
        return df_results

    def run_cover_set_no_cat_scopus(self, utils=None):
        cats_dict = utils.load_obj('no_cat_scopus_to_wos_categories_for_group_mapping')
        journals_set=set()
        for k,v in cats_dict.items():
            journals_set.update(v)
        count=0
        for k,v in cats_dict.items():
            count+=len(v)
        df_results = pd.DataFrame(
            columns=['Category', 'Num journals', 'Num matching cats', 'Min cover set Greedy', 'Min Cover set ILP'])
        journals_df=pd.DataFrame(columns=['Journal title'])
        journals_df['Journal title']=list(journals_set)
        start_time = datetime.now()
        cover_set_greedy = cs.cover_set_greedy(journals=journals_df, cats_dict=cats_dict)
        end_time = datetime.now()
        print(end_time - start_time)
        greedy_cover_set_size = len(cover_set_greedy)
        const_arr = cs.build_array(journals=journals_df, cats_dict= cats_dict)
        ilp_model = cs.build_model(const_arr)
        status, objective = cs.run_model(ilp_model)
        print('All journals, all categories, status {}, objective {}'.format(status, objective))
        record = {'Category': 'All', 'Num journals': len(journals_set), 'Num matching cats': len(cats_dict),
                  'Min cover set Greedy': greedy_cover_set_size, 'Min Cover set ILP': int(objective)}
        return record


if __name__ == '__main__':
    cs=CoverSet()
    utils=Utils(path=path)
    vis=Visualization()

    df=cs.run_cover_set_per_category_wos()
    utils.save_obj(df,'cover_set_wos_to_scopus')
    exit(0)
    df=utils.load_obj('cover_set_wos_to_scopus')
    #
    record=cs.run_cover_set_no_cat_wos()
    df=df.append(record, ignore_index=True)
    print(df)
    utils.save_obj(df,'cover_set_wos_to_scopus_full')
    df_cover_set_wos_to_scopus_full=utils.load_obj('cover_set_wos_to_scopus_full')
    all_wos=df_cover_set_wos_to_scopus_full.iloc[-1]

    # sorted_df_by_num_journals = df.sort_values(by='Num journals')
    # vis.plt_coverset_size(df_cover_set_wos_to_scopus_full[0:-1],'Cover set size by Number of journals', 'Cover set size by Number of corresponding categories')
    # sorted_df_by_num_cats = df.sort_values(by='Num matching cats')
    # vis.plt_coverset_size(sorted_df_by_num_cats[0:-1], 'Cover set size by number of categories', by_journals=False)

    # df=cs.run_cover_set_per_category_scopus()
    # utils.save_obj(df,'cover_set_scopus_to_wos')
    # df_cover_set_scopus_to_wos=utils.load_obj('cover_set_scopus_to_wos')
    # record=cs.run_cover_set_no_cat_scopus()
    # df_cover_set_scopus_to_wos = df_cover_set_scopus_to_wos.append(record, ignore_index=True)
    # utils.save_obj(df_cover_set_scopus_to_wos,'cover_set_scopus_to_wos_full')

    df_cover_set_scopus_to_wos_full = utils.load_obj('cover_set_scopus_to_wos_full')
    all_scopus=df_cover_set_scopus_to_wos_full.iloc[-1]
    print('All {} journals covered by {}. Number of categories {}. number of journals {}. Min cover set {}'.format('WOS','Scopus',all_wos['Num matching cats'],all_wos['Num journals'],all_wos['Min Cover set ILP']))
    print('All {} journals covered by {}. Number of categories {}. number of journals {}. Min cover set {}'.format('Scopus','WOS',all_scopus['Num matching cats'],all_scopus['Num journals'],all_scopus['Min Cover set ILP']))

    vis.plt_coverset_size(df_cover_set_wos_to_scopus_full[0:-1], df_cover_set_scopus_to_wos_full[0:-1],'Cover set size by Number of journals', 'Cover set size by Number of corresponding categories', extract_low=20)


    # record=cs.run_cover_set_no_cat_scopus()




    # cs.calc_cover_set_ILP_pulp()

        # record={'Category': wos_category,'Num journals': len(journals),'Num matching cats','Min cover set Greedy','Min Cover set ILP'}
            # break;
    # cs.test()
