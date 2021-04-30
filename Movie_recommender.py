# -*- coding: utf-8 -*-
"""
@author: Anshuman Raj
"""
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
corrMatrix = userRatings.corr()
myRatings = userRatings.loc[0].dropna()
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that the user have rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scaleing its similarity by how well this movie was rated
    sims = sims.map(lambda x: x * myRatings[i])
    # Added the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)
    
#Glance at our results so far:
print ("sorting...")
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))

simCandidates = simCandidates.groupby(simCandidates.index).sum()
simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)