import facenet
import numpy as np

embeddings1 = np.array([[1, 0], [0, 1]])
embeddings2 = np.array([[1, 1], [0, 1]])

dist = facenet.distance(embeddings1, embeddings2, 0)
print(dist)