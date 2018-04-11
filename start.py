import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

file = r'/home/abha/SE-Project-Twitzee-datasets/Movies_Datasets/movies_metadata.csv'
dataset = pd.read_csv(file, low_memory=False)
# print(dataset.columns.values)

class SimpleRecommender:

    def weighted_rating(self, row, m, C):
        v = row['vote_count']
        R = row['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)    #IMDb formula

    def simpleRecommender(self, limit):
        C = dataset['vote_average'].mean()
        m = dataset['vote_count'].quantile(limit)
        copy_filtered = dataset.copy().loc[dataset['vote_count'] >= m]
        copy_filtered['score'] = copy_filtered.apply(lambda row: self.weighted_rating(row, m, C), axis=1)
        copy_filtered = copy_filtered.sort_values('score', ascending=False)
        print(copy_filtered[['title', 'vote_count', 'vote_average', 'score']].head(15))

    def __init__(self, l):
        self.simpleRecommender(limit=l)

# class ContentBasedRecommender:
#
#     title = ''
#     tfidf = TfidfVectorizer(stop_words='english')
#     dataset['overview'] = dataset['overview'].fillna('')
#     tfidf_matrix = tfidf.fit_transform(dataset['overview'])
#     cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#     indices = pd.Series(dataset.index, index=dataset['title']).drop_duplicates()
#
#     def getRecommendations(self, title, cosine_sim=cosine_sim):
#         idx = self.indices[title]
#
#
#     def __init__(self, t):
#         self.title = t




def main():
    print("Simple Recommender:")
    SimpleRecommender(l=0.90)

if __name__ == '__main__':
    main()


