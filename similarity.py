import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm

def embed(input):
    return model(input)

# Step-1: Importing required libraries and loading the TensorFlow Hub module for the Universal Sentence Encoder.
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

# Step-2: Reading Data
data = pd.read_csv("Text_Similarity.csv")

# Step-3: Encoding text to vectors
def calculate_similarity(data):
    ans = []
    for i in range(len(data)):
        # Extracting sentence pairs
        messages = [data['text1'][i], data['text2'][i]]
        # Converting sentence pair to vector pair using Universal Sentence Encoder
        message_embeddings = embed(messages)
        # Converting tensor to numpy array
        a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
        # Calculating cosine similarity between the two vectors
        cos_sim = dot(a[0], a[1]) / (norm(a[0]) * norm(a[1]))
        ans.append(cos_sim)
    return ans

# Step-4: Finding Cosine similarity
ans = calculate_similarity(data)

# Step-5: Creating DataFrame and saving to CSV
Answer = pd.DataFrame(ans, columns=['Similarity_Score'])
data = data.join(Answer)

# Shifting Similarity_Score range from [-1,1] to [0,2] and then normalizing it to [0,1]
data['Similarity_Score'] = data['Similarity_Score'] + 1
data['Similarity_Score'] = data['Similarity_Score'] / data['Similarity_Score'].abs().max()

# Create a 'Unique_ID' column
data.insert(0, 'Unique_ID', range(1, len(data) + 1))

Submission_task = data[['Unique_ID', 'Similarity_Score']]
Submission_task.set_index("Unique_ID", inplace=True)
Submission_task.to_csv('Submission_Task.csv')

print("File 'Submission_Task.csv' created successfully.")

# Optionally, if running in an environment that supports file downloads
# from google.colab import files
# files.download('Submission_Task.csv')