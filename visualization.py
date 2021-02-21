import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
import numpy as np
from textwrap import wrap
from utils import Utils
import math



import csv
import os

class Visualization():
    font = {
        # 'family': 'Ariel',
            'weight': 'normal',
            'size': 12}

    plt.rc('font', **font)

    def plt_match_by_threshold(self,df_w_to_s,df_s_to_w, title):
        f = plt.figure()
        ax = f.gca()
        # plt.title(title, color='black')
        df_w_to_s.plot(ax=ax, legend=False, title=title, xticks=df_w_to_s.index, rot=45, ylabel=df_w_to_s.columns[0], color='blue')
        df_s_to_w.plot(ax=ax, legend=False, title=title, xticks=df_s_to_w.index, rot=45, ylabel=df_s_to_w.columns[0], color='green')

        # ax.set_xticks(df.index)
        # ax.set_ylabel(df.columns[0])
        ax.legend(['Scopus category for WOS category','WOS category for Scopus category'])
        plt.show()

    def plt_coverset_size(self,df1,df2,title1, title2, extract_low=None, extract_high=None):
        fig, (ax1, ax2)=plt.subplots(1, 2)

        sorted_df1 = df1.sort_values(by='Num journals')
        sorted_df2 = df2.sort_values(by='Num journals')

        if isinstance(extract_low, int):
            sorted_df1=sorted_df1[sorted_df1['Num journals'] <= extract_low]
            sorted_df2=sorted_df2[sorted_df2['Num journals'] <= extract_low]

        if isinstance(extract_high, int):
            sorted_df1 = df1.sort_values(by='Min Cover set ILP')
            sorted_df2 = df2.sort_values(by='Min Cover set ILP')
            sorted_df1=sorted_df1[sorted_df1['Min Cover set ILP'] >= extract_high]
            sorted_df2=sorted_df2[sorted_df2['Min Cover set ILP'] >= extract_high]



        # ys1 = sorted_df1['Min cover set Greedy'].values
        ys1 = sorted_df1['Min Cover set ILP'].values
        ys2 = sorted_df2['Min Cover set ILP'].values
        xs1=sorted_df1['Num journals'].values
        xs2 = sorted_df2['Num journals'].values

        ax1.scatter(xs1, ys1,color='blue', label='Scopus categories min \n cover set of WOS categories')
        ax1.scatter(xs2,ys2, color='green', label='WOS categories min \n cover set of Scopus categories')
        ax1.set_ylabel('Min cover set size')
        ax1.set_xlabel('Number of journals')



        ax1.set_title(title1)

        sorted_df1 = df1.sort_values(by='Num matching cats')
        sorted_df2 = df2.sort_values(by='Num matching cats')

        if isinstance(extract_low, int):
            sorted_df1=sorted_df1[sorted_df1['Num matching cats'] <= extract_low-10]
            sorted_df2=sorted_df2[sorted_df2['Num matching cats'] <= extract_low-10]

        if isinstance(extract_high, int):
            sorted_df1 = df1.sort_values(by='Min Cover set ILP')
            sorted_df2 = df2.sort_values(by='Min Cover set ILP')
            sorted_df1 = sorted_df1[sorted_df1['Min Cover set ILP'] >= extract_high]
            sorted_df2 = sorted_df2[sorted_df2['Min Cover set ILP'] >= extract_high]




        ys1 = sorted_df1['Min Cover set ILP'].values
        ys2 = sorted_df2['Min Cover set ILP'].values
        xs1 = sorted_df1['Num matching cats'].values
        xs2 = sorted_df2['Num matching cats'].values

        ax2.scatter(xs1, ys1, color='blue', label='Scopus categories min \n cover set of WOS categories')
        ax2.scatter(xs2, ys2, color='green', label='WOS categories min \n cover set of Scopus categories')
        ax2.set_ylabel('Min cover set size')
        ax2.set_xlabel('Num corresponding categories')

        ax2.set_title(title2)

        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.80, 0.55))
        plt.show()

    def plt_histogram_cats(self, df, title):
        xticks = np.arange(df['num categories'].values.min(),
                           df['num categories'].values.max() + 1)
        ax=df.hist(column='num categories', grid=False, zorder=2, rwidth=0.9)
        ax = ax[0]

        for x in ax:
            # Despine
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
            # x.spines['left'].set_visible(False)
            # x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
            #               labelleft="on")
            x.set_xticks(xticks)
            # x.set_xticklabels(df.index, rotation=45)

        plt.xlabel('Number of categories')
        plt.ylabel('Number of Journals')
        plt.title(title)
        plt.show()


    def plt_histogram_cats_intersection(self, df, title):
        xticks = np.arange(df['num intersecting categories'].values.min(),
                           df['num intersecting categories'].values.max() + 1)

        ax=df.hist(column='num intersecting categories', bins=20, grid=False, zorder=1, rwidth=0.9, orientation='horizontal')
        ax = ax[0]

        for x in ax:
            # Despine
            x.spines['right'].set_visible(False)
            x.spines['top'].set_visible(False)
            # x.spines['left'].set_visible(False)
            # x.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off",
            #               labelleft="on")
            # x.set_xticks(xticks)
            # x.set_xticklabels(df.index, rotation=45)

        plt.xlabel('Number of categories')
        plt.ylabel('Number of intersecting categories')
        plt.title(title)
        plt.show()


    def get_subplots_for_venn(self):
        figure, axes = plt.subplots(4, 3)
        return figure, axes

    def create_venn_diagrams(self,subsets, labels, ax):
        venn2(subsets=subsets, set_labels=labels, set_colors=('purple', 'skyblue'), alpha=0.7, ax=ax)
        # plt.show()

    def plt_groups_data(self, groups, plt_by, title):
        fig, axs = plt.subplots(1, 1, facecolor='w')
        ax1 = axs
        ax6 = ax1.twinx()
        ax1.yaxis.tick_left()
        ax1.tick_params(axis='y', color='white', labelsize='14')
        ax6.yaxis.tick_right()
        ax6.tick_params(axis='y', color='white', labelsize='14')
        ylabel=plt_by
        mean_score = groups[plt_by].mean()
        highest_score = groups[plt_by].max()
        median_score = groups[plt_by].median()
        percentiles_10 = groups[plt_by].quantile([0.1])
        percentiles_30 = groups[plt_by].quantile([0.3])
        percentiles_75 = groups[plt_by].quantile([0.75])
        percentiles_90 = groups[plt_by].quantile([0.9])
        print('highest JIF {}'.format(highest_score))
        print('mean JIF {}'.format(mean_score))
        print('median JIF {}'.format(median_score))
        self.plot_category_groups(ax1, mean_score, ylabel=plt_by, label='Mean')
        self.plot_category_groups(ax1, median_score, ylabel=plt_by, label='Median')
        self.plot_category_groups(ax6, highest_score, ylabel='Max '+plt_by, label='Max', linestyle='dashed')
        # self.plot_category_groups(ax1, percentiles_10.reset_index(1)[plt_by], ylabel=plt_by, label='10th percentile')
        self.plot_category_groups(ax1, percentiles_30.reset_index(1)[plt_by], ylabel=plt_by, label='30th percentile')
        # self.plot_category_groups(ax1, percentiles_75.reset_index(1)[plt_by], ylabel=plt_by, label='75th percentile')
        self.plot_category_groups(ax1, percentiles_90.reset_index(1)[plt_by], ylabel=plt_by, label='90th percentile')
        ax1.set_xlabel('Number of Categories')
        plt.title(title)
        fig.legend(loc=(0.7, 0.55))
        plt.show()


        # self.plot_category_groups(ax1, groups, ylabel=ylabel, label=label)

    def plot_category_groups(self,axis,groups, ylabel, label, linestyle=None):
        if linestyle!=None:
            axis.scatter(groups.index, groups.values, label=label, marker= '*', color='cyan')
        else:
            axis.scatter( groups.index, groups.values, label=label)
        axis.set_ylabel(ylabel)

    def plt_show_and_title(self, title):
        plt.title(title)
        # plt.legend()
        plt.show()