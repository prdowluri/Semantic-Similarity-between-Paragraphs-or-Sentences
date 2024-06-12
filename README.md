# **Text-Similarity-Analyzer**

## **Part A: Task Description:**
The task at hand involves evaluating the semantic similarity between two paragraphs. Semantic Textual Similarity (STS) measures the extent to which two pieces of text convey similar meanings. STS entails assesses the degree to which two sentences are semantically equivalent to each other. Our task is to involves the producing real-valued similarity scores for sentence pairs.

## **Description of the Data:**
The dataset comprises pairs of paragraphs randomly selected from a larger raw dataset. These paragraphs may or may not exhibit semantic similarity. Participants are tasked with predicting a value ranging from 0 to 1, which indicates the degree of similarity between each pair of text paragraphs. A score of
1 means highly similar
0 means highly dissimilar

## **Approach to solve this problem:**
To solve this Natural Language Processing (NLP) problem, the initial step involves text embedding, a pivotal aspect in building deep learning models. Text embedding transforms textual data (such as sentences) into numerical vectors.

Once the sentences are converted into vectors, we can calculate how close these vectors are based on the cosine similarity.

We are not converting just based on keyword. Here, we need to concentrate the context and meaning of the text.

To address this, we leverage the Universal Sentence Encoder (USE). This encoder translates text into higher-dimensional vectors, which are ideal for our semantic similarity task. The pre-trained Universal Sentence Encoder (USE) is readily accessible through TensorFlow Hub, providing a robust solution for our needs.
