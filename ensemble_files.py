import pandas as pd
import sys

first_file = sys.argv[1]
second_file = sys.argv[2]
outfile = sys.argv[3]

def ensemble(first_file, second_file):
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    # assuming first column is `prediction_id` and second column is
    # `prediction`
    full_df = pd.concat([first_df, second_df], axis=1)
    print(full_df.head(2).mean())
    # prediction = first_df.columns[0]
    # # correlation
    # print("Finding correlation between: %s and %s" % (first_file, second_file))
    # print("Column to be measured: %s" % prediction)
    # print("Pearson's correlation score: %0.5f" %
    #       first_df[prediction].corr(second_df[prediction], method='pearson'))
    # print("Kendall's correlation score: %0.5f" %
    #       first_df[prediction].corr(second_df[prediction], method='kendall'))
    # print("Spearman's correlation score: %0.5f" %
    #       first_df[prediction].corr(second_df[prediction], method='spearman'))


ensemble(first_file, second_file, outfile)
