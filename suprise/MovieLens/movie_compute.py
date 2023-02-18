import pandas as pd

data = pd.read_csv('ratings.csv')
result = data.groupby('rating')['rating'].agg(['count'])
print(result)


#movie_ratings = pd.DataFrame(data.groupby('movieId')['rating'].mean())
#movie_ratings.columns = ['movieId', 'rating']
print(movie_ratings)
#movie_ratings.to_csv('movie_ratings.csv')
#print( )

"""
movie_ratings = {}
for i in range(ratings):
	movieId = ratings['movieId']

	if movieId not in movie_ratings:

	movId in ratings['userId']:

	print(userId)
#userId,movieId,rating,timestamp
"""