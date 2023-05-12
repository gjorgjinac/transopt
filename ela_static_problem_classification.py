import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.metrics import classification_report
import sys


dimension=int(sys.argv[1])
ela=pd.read_csv(f'ela_data/lhs_samples_dimension_{dimension}_50_samples.csv',index_col=[0]).set_index(['problem_id','instance_id'])
result_dir='results_ela'
ela_groups=list(set([x.split('.')[0] for x in ela.columns])) + ['all']



def preprocess_ela(train,test):
    train=train.select_dtypes(exclude=['object'])

    train=train.replace([np.inf, -np.inf], np.nan)
    test=test.replace([np.inf, -np.inf], np.nan)
    count_missing={column:train[column].isna().sum() for column in train.columns}
    count_missing=pd.DataFrame([count_missing]).T
    count_missing.columns=['missing']
    count_missing['missing_percent']=count_missing['missing'].apply(lambda x: x/train.shape[0])
    
    columns_to_keep=list(count_missing.query('missing_percent<0.1').index)
    print('Keeping columns', columns_to_keep)
    train=train[columns_to_keep]
    test=test[columns_to_keep]
    print(train)
    train = train.fillna(train.mean()).astype(np.float32)
    test = test.fillna(train.mean()).astype(np.float32)
    return train,test




for group in ela_groups:
    for fold in range(0,10):
        run_name=f'dim_{dimension}_fold_{fold}_features_{group}'
        if os.path.isfile(f'{result_dir}/{run_name}_classification_report.csv'):
            continue
        train_instance_ids, val_instance_ids, test_instance_ids = [list(pd.read_csv(f'folds/problem_classification_999_instances/{split_name}_{fold}.csv',index_col=[0])['0'].values) for split_name in ['train','val','test']]
        if group=='all':
            group_features=ela.columns
        else:
            group_features=list(filter(lambda c: c.startswith(f'{group}.'), ela.columns))
        
        train_X, val_X, test_X = [ela.query('instance_id in @split_ids')[group_features] for split_ids in [train_instance_ids, val_instance_ids, test_instance_ids]]

        train_y, val_y, test_y = [d.reset_index()['problem_id'] for d in [train_X, val_X, test_X]]

        train_X,test_X=preprocess_ela(train_X,test_X)

        clf = RandomForestClassifier()
        print(train_X.shape)
        print(train_y.shape)
        clf.fit(train_X, train_y)

        

        preds = clf.predict(test_X)
        report_dict = classification_report(test_y, preds,  output_dict=True)
        report_df = pd.DataFrame(report_dict)
        report_df.to_csv(f'{result_dir}/{run_name}_classification_report.csv')

        #save_feature_importance(run_name, clf, dimension, iteration_min, iteration_max, result_dir,feature_names)

        test_predictions=pd.DataFrame(list(zip(test_y.values,preds)), index=test_X.index, columns=['y','preds'])
        test_predictions.to_csv(f'{result_dir}/{run_name}_test_preds.csv', compression='zip')
        feature_importance_df=pd.DataFrame(list(clf.feature_importances_)).T
        feature_importance_df.columns=train_X.columns
        feature_importance_df.to_csv(f'{result_dir}/{run_name}_feature_importance.csv', compression='zip')
