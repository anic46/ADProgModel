#!/usr/bin/env python
import argparse
import numpy as np
import pandas as pd

import src.models.minRNN.misc as misc 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spreadsheet', required=True)
    parser.add_argument('--cogsc', required=True)
    parser.add_argument('--metrics', choices=['MAE', 'MSE'], required=True)
    parser.add_argument('--cols', required=True)
    parser.add_argument('--vals', required=True)
    parser.add_argument('--year', type=int, default=0)

    return parser.parse_args()

def MAE(df, cogsc):
	return np.abs(df[cogsc] - df[cogsc+'_pred']).mean()

def MSE(df, cogsc):
	return ((df[cogsc] - df[cogsc+'_pred'])**2).mean()

METRIC_DICT = {'MAE': MAE, 'MSE': MSE}

def main(args):
	ans = 0
	for i in range(5):
		data_df = pd.read_csv(args.spreadsheet+str(i)+'.csv')
		
		cols = misc.load_feature(args.cols)
		vals = misc.load_feature(args.vals)
		assert len(cols)==len(vals)
		for c, v in zip(cols, vals):
			if c in data_df.columns:
				data_df = data_df[data_df[c]==v]
		if args.year:
			data_df = data_df[data_df['Years']==args.year]

		ans += METRIC_DICT[args.metrics](data_df, args.cogsc)
	print(args.metrics, ': {:.2f}'.format(ans/5))


if __name__ == '__main__':
    main(get_args())