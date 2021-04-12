import flask
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import json
from flask import Flask, render_template, flash, request
from model import model_prediction

#df_recom = pd.read_csv('recommendationdata.csv')
#df_all = pd.read_csv('sample30.csv')
#df = pd.read_csv('recompivot.csv', index_col=0)

# Create the application.
app = flask.Flask(__name__)


@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return flask.render_template('index.html')



def item_similarity(ratings, epsilon=1e-9):
    # epsilon -> for handling dived-by-zero errors
    sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

def predict_item(ratings, similarity):
    return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

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
    topk=recommendations.merge(original_ratings_df, right_on = 'name',left_on='name').drop_duplicates(
    ['name'])[['name']]

    return topk

@app.route('/predict', methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        username=request.form['username']
        print(username)

        df_recom = pd.read_csv('recommendationdata.csv')
        df_all = pd.read_csv('sample30.csv')
        #df = pd.read_csv('recompivot.csv', index_col=0)
    

        ratingsd=df_recom.pivot(index='name',columns= 'reviews_username',values='rating').fillna(0)
        
        print(ratingsd.dtypes)

        item_sim = item_similarity(ratingsd)

        item_prediction = predict_item(ratingsd, item_sim)

        svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
        r_mat_tr=svd.fit_transform(ratingsd) 
        print(svd.explained_variance_ratio_)  
        print(svd.explained_variance_ratio_.sum())

        #pm=pd.DataFrame(cosine_similarity(r_mat_tr))
        #pm.head()
        ctrain = cosine_similarity(r_mat_tr)

        df_recom = df_recom.sort_values(by='rating')
        df_recom = df_recom.reset_index(drop=True)
        count_users = df_recom.groupby("reviews_username", as_index=False).count()


        count = df_recom.groupby("name", as_index=False).mean()

        items_df = count[['name']]
        users_df = count_users[['reviews_username']]

        df_clean_matrix = df_recom.pivot(index='name', columns='reviews_username', values='rating').fillna(0)
        df_clean_matrix = df_clean_matrix.T
        R = (df_clean_matrix).values
        

        user_ratings_mean = np.mean(R, axis = 1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        R_demeaned

        U, sigma, Vt = svds(R_demeaned)

        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_clean_matrix.columns)
        preds_df['reviews_username'] = users_df
        preds_df.set_index('reviews_username', inplace=True)


        df= preds_df

        s = recommend_it(df, items_df, df_recom, 20,username)
        
        result = model_prediction(s)

        print(result)
    #return flask.render_template('main.html', title="Results", tables = json.dumps(result))
    return flask.render_template('main.html',tables=[result], titles = [username, ' Product Recommendations'])
    #return flask.render_template('index.html')



if __name__ == '__main__':
    app.debug=True
    app.run(debug=True)
