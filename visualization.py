import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from utils import Utils
import math



import csv
import os

class Visualization():



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