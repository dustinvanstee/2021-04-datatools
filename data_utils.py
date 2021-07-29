import sys,os
import numpy as np
import pandas as pd
import seaborn as sns

def nprint(mystring) :
    print("# {:20s}: {}".format(sys._getframe(1).f_code.co_name,mystring))
    
def quick_overview_1d(df) :
    '''
    Prints useful statistics about the dataframe ...
    '''
    df_meta = pd.DataFrame(df.dtypes, columns=["dtype"])
    nprint("There are " + str(len(df)) + " observations in the dataset.")
    nprint("There are " + str(len(df.columns)) + " variables in the dataset.")

    nprint("\n\n")
    nprint("\n****************** Histogram of data types  *****************************\n")
    nprint("use df.dtypes ...")
    print(df_meta['dtype'].value_counts())

    # Cardinality Report
    nprint("\n\n")
    nprint("\n****************** Generating Cardinality Report (all types) *****************************\n")
    #tmpdf = df.select_dtypes(include=['object'])
    # Cardinality report
    card_count = []
    card_idx = []
    for c in df.columns :
        #print("{} {} ".format(c, len(df[c].value_counts())))
        card_count.append(len(df[c].value_counts()))
        card_idx.append(c)
    card_df = pd.DataFrame(card_count, index =card_idx, 
                                          columns =['cardinality']) 
    df_meta = df_meta.join(card_df, how="outer")
    
    nprint("\n****************** Generating NaNs Report *****************************\n")
    nan_df = pd.DataFrame(df.isna().sum(), columns=['nan_count'])
    df_meta = df_meta.join(nan_df, how="outer")
    # Add pct missing
    df_meta['pct_missing'] = 100 * (df_meta['nan_count'] / len(df))

    nprint("\n******************Generating Descriptive Statistics (numerical columns only) *****************************\n")
    #print(" running df.describe() ....")
    desc_df = df.describe().transpose()
    df_meta = df_meta.join(desc_df, how="outer")

    df_meta = df_meta.sort_values(by=['dtype'],ascending=False)
    df_meta = df_meta.transpose()

    cat_cols = df.select_dtypes(include=['object']).columns
    num_cols = df.select_dtypes(include=['float64','int']).columns

    return (df_meta[cat_cols].iloc[0:4,:], df_meta[num_cols])


def quick_overview_1d(df) :
    
    nprint("There are " + str(len(df)) + " observations in the dataset.")
    nprint("There are " + str(len(df.columns)) + " variables in the dataset.")

    nprint("\n\n")
    nprint("\n****************** Categorical vs Numerical  *****************************\n")
    dt = pd.DataFrame(df.dtypes)
    dt.columns = ['type']
    nprint("use df.dtypes ...")
    print(dt['type'].value_counts())

    # Cardinality Report
    nprint("\n\n")
    nprint("\n****************** Cardinality Report  *****************************\n")
    tmpdf = df.select_dtypes(include=['object'])
    # Cardinality report
    for c in tmpdf.columns :
        print("{} {} ".format(c, len(tmpdf[c].value_counts())))

    nprint("\n******************Dataset Descriptive Statistics (numerical columns only) *****************************\n")
    print(" running df.describe() ....")
    return df.describe()


def quick_overview_2d(loan_df, cols) :
    nprint("There are " + str(len(loan_df)) + " observations in the dataset.")
    nprint("There are " + str(len(loan_df.columns)) + " variables in the dataset.")

    df = loan_df[cols]
    corr_df = df.corr()
    # plot the heatmap
    sns.set_style(style = 'white')
    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(corr_df, cmap=cmap,
        xticklabels=corr_df.columns,
        yticklabels=corr_df.columns,vmin=-1.0,vmax=1.0) 
