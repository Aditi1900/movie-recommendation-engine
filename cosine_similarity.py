from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)

print(count_matrix.toarray())  # Output explanation: # The vectors are plotted according to their word repetition, eg.
                               # first vector is [2 1] because London occurs twice and paris once in that sentence and
                               # similarly for the other sentence

similarity_score = cosine_similarity(count_matrix)
print(similarity_score)  # Output explanation: the first text is similar to the first text (itself) by 100 percent so 1
                         # first sentence is similar to second sentence by cos theta of 0.8,second is similar to the
                         # first one by 0.8 and second is similar to itself by 0.8 too
                         # Because cos of 0 degrees is 1, and here theta is 36.87 for the distance between the vectors
                         # "london" and "paris" to be 0.8
