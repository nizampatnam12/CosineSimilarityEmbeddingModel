import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'  # Replace with your actual API key

def generate_embeddings(text):
    """
    Function to generate embeddings for a given text using OpenAI's API.
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-3-small",  # Correct model for embeddings
            input=text
        )
        # Extract embeddings from the response
        embeddings = response['data'][0]['embedding']
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def quantize_embeddings(embeddings, dtype=np.int8):
    """
    Quantize the embeddings to a lower precision (e.g., int8).
    
    Parameters:
    - embeddings: The embeddings to quantize (list or numpy array).
    - dtype: The data type to quantize to (default: np.int8).
    
    Returns:
    - Quantized embeddings in the specified dtype.
    """
    # Convert embeddings to numpy array for easier manipulation
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Normalize embeddings to a 0-255 range for int8 quantization (optional scaling step)
    min_val = np.min(embeddings_array)
    max_val = np.max(embeddings_array)
    
    # Scale to [0, 255] range
    scaled_embeddings = 255 * (embeddings_array - min_val) / (max_val - min_val)
    
    # Quantize to the desired dtype (int8 in this case)
    quantized_embeddings = scaled_embeddings.astype(dtype)
    
    # Return quantized embeddings
    return quantized_embeddings

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Function to calculate the cosine similarity between two embeddings.
    """
    # Reshape embeddings to 2D array for compatibility with sklearn's cosine_similarity function
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    
    # Calculate cosine similarity using sklearn's cosine_similarity function
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

# Example usage
if __name__ == "__main__":
    input_text_1 = "Dog"
    input_text_2 = "Cat"
    
    # Generate embeddings for the input texts
    embedding1 = generate_embeddings(input_text_1)
    embedding2 = generate_embeddings(input_text_2)
    
    if embedding1 and embedding2:
        # Quantize the embeddings before proceeding
        quantized_embedding1 = quantize_embeddings(embedding1)
        quantized_embedding2 = quantize_embeddings(embedding2)
        
        # Calculate cosine similarity
        similarity = calculate_cosine_similarity(quantized_embedding1, quantized_embedding2)
        print(f"Cosine Similarity between '{input_text_1}' and '{input_text_2}': {similarity}")
    else:
        print("Error generating embeddings.")
