import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item ratings
data = {'User': [1, 1, 1, 2, 2, 3, 3],
        'Item': [1, 2, 3, 2, 3, 1, 3],
        'Rating': [5, 4, 2, 5, 3, 4, 1]}
df = pd.DataFrame(data)

# Create a pivot table
pivot_table = df.pivot(index='User', columns='Item', values='Rating').fillna(0)

# Calculate similarity
similarity = cosine_similarity(pivot_table)
print(similarity)
