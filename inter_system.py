from utils import Utils
from itertools import chain,combinations
import pandas as pd



class InterSystem:
    def cover_set_greedy(self, journals, cats_dict):
        sorted_cats = sorted(cats_dict, key=lambda k: len(cats_dict[k]))
        universe = journals
        universe['Journal title'] = universe.apply(lambda row: pd.Series(row['Journal title'].lower()), axis=1)
        cover = set()
        while not len(universe) == 0:
            current_cat_name = sorted_cats.pop()
            current_journals = cats_dict[current_cat_name]
            # current_cat=cats_dict.pop()
            current_universe = universe[~universe['Journal title'].isin(current_journals)]
            if len(current_universe) < len(universe):
                cover.add(current_cat_name)
                universe = current_universe
        return cover

    def cover_set_brute_force(self, journals, cats_dict):
        universe = journals
        cat_names = list(cats_dict.keys())
        cats_power_group = chain.from_iterable(combinations(cat_names, r) for r in range(3, len(cat_names) + 1))
        print(cats_power_group)
        for subset in cats_power_group:
            current_universe = universe
            current_cover_set_size = len(subset)
            # print(current_cover_set_size)
            for current_set in subset:
                current_journals = cats_dict[current_set]
                current_universe = current_universe[~current_universe['Journal title'].isin(current_journals)]
            if len(current_universe) == 0:
                cover = subset
                print(cover)
                break
        return cover

    def run_cover_set_per_category(self, utils):
        df = utils.load_obj('wos_to_scopus_categories_for_group_mapping')
        cover_set_df = pd.DataFrame(columns=['Category', 'Num journals', 'Cover set size'])
        for wos_category, row in df.iterrows():
            sco_cats_dict = row['scopus_categories']
            if isinstance(sco_cats_dict, float):
                print('skipping cat {}'.format(wos_category))
                continue
            journals = row['journals']
            print('number cats {}, num journals {}'.format(len(sco_cats_dict), len(journals)))
            cover_set = self.cover_set_greedy(journals=journals, cats_dict=sco_cats_dict)
            cover_set_size = len(cover_set)
            print('cover set size for greeedy alg {}'.format(cover_set_size))
            if cover_set_size <= 200:
                record = {'Category': wos_category, 'Num journals': len(journals), 'Cover set size': cover_set_size}
                cover_set_df = cover_set_df.append(record, ignore_index=True)
                print('{} greedy size {}'.format(wos_category, cover_set_size))
            else:
                cover_set_2 = self.cover_set_brute_force(journals=journals, cats_dict=sco_cats_dict)
                cover_set_2_size = len(cover_set_2)
                record = {'Category': wos_category, 'Num journals': len(journals), 'Cover set size': cover_set_2_size}
                cover_set_df = cover_set_df.append(record, ignore_index=True)
                if cover_set_2_size > cover_set_size:
                    print('Error for {} greedy size {}, brute force {}'.format(wos_category, cover_set_size,
                                                                               cover_set_2_size))
                else:
                    if cover_set_2_size == cover_set_size:
                        print('no change in size for {} greedy size {}, brute force {}'.format(wos_category,
                                                                                               cover_set_size,
                                                                                               cover_set_2_size))
                    else:
                        print('found smaller cover set for {} greedy size {}, brute force {}'.format(wos_category,
                                                                                                     cover_set_size,
                                                                                                     cover_set_2_size))
        return cover_set_df










