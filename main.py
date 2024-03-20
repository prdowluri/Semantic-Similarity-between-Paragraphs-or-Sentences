import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
from numpy import dot
from numpy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

# Define the FastAPI app
app = FastAPI()

# Step-1: Loading the TensorFlow Hub module for the Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

# Step-3: Encoding text to vectors
def embed(input):
    return model(input)

# Step-4: Function to calculate similarity
def calculate_similarity(text1: str, text2: str):
    # Converting sentence pair to vector pair using Universal Sentence Encoder
    message_embeddings = embed([text1, text2])
    # Converting tensor to numpy array
    embeddings_array = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    # Calculating cosine similarity between the two vectors
    cos_sim = dot(embeddings_array[0], embeddings_array[1]) / (norm(embeddings_array[0]) * norm(embeddings_array[1]))
    # Shifting Similarity_Score range from [-1,1] to [0,2] and then normalizing it to [0,1]
    similarity_score = (cos_sim + 1) / 2
    return similarity_score

# Define request body model
class TextPair(BaseModel):
    text1: str
    text2: str

# Define endpoint for calculating similarity score
@app.post("/calculate_similarity/")
async def get_similarity_score(text_pair: TextPair):
    similarity_score = calculate_similarity(text_pair.text1, text_pair.text2)
    return {"similarity_score": similarity_score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)