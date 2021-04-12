import flask
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import json
from flask import Flask, render_template, flash, request
import pickle



df_recom = pd.read_csv('recommendationdata.csv')
#df_all = pd.read_csv('sample30.csv')
#df = pd.read_csv('recompivot.csv', index_col=0)

model = pickle.load(open('model.pkl', 'rb'))


def recommend_it(predictions_df, itm_df, original_ratings_df, num_recommendations=10,ruserId='00dog3'):
    
    # Get and sort the user's predictions
    sorted_user_predictions = predictions_df.loc[ruserId].sort_values(ascending=False)
    
    # Get the user's data and merge in the item information.
    user_data = original_ratings_df[original_ratings_df.reviews_username == ruserId]
    user_full = (user_data.merge(itm_df, how = 'left', left_on = 'name', right_on = 'name').
                     sort_values(['rating'], ascending=False)
                 )

    print ('User {0} has already purchased {1} items.'.format(ruserId, user_full.shape[0]))
    print ('Recommending the highest {0} predicted  items not already purchased.'.format(num_recommendations))
    
    # Recommend the highest predicted rating items that the user hasn't bought yet.
    recommendations = (itm_df[~itm_df['name'].isin(user_full['name'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'name',
               right_on = 'name').
         rename(columns = {ruserId: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )
    topk=recommendations.merge(original_ratings_df,left_index = True, right_on = 'name',left_on='name').drop_duplicates(
    ['name'])[['name']]

    return topk

def model_prediction(s):
    print(s)

    d = df_all[df_all['name'].isin(s['name'])]
        #result = s.to_json(orient="split")

    filter_data = pd.DataFrame({'Per':d.groupby(['name','user_sentiment']).size()})
        # Change: groupby state_office and divide by sum
    filter_data_per = filter_data.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum()))

    s = filter_data.groupby(['name'])['Per'].sum().reset_index()
    s =s[s['Per']>100]

    filter_data_per = filter_data_per.reset_index()
    filter_data_per = filter_data_per[filter_data_per['user_sentiment'] == 'Positive'].sort_values(by=['Per'], ascending=False)

    filter_data_per = filter_data_per['name'].head()

    result = filter_data_per.to_json(orient="split")
    result = json.loads(result)

    return result
