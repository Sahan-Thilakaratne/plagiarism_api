from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load model specialized for paraphrase detection
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def compute_similarity(text1, text2):
    # Encode the two sentences
    embeddings = model.encode([text1, text2])

    # Compute cosine similarity between the two embeddings
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    return similarity


def classify_similarity(similarity_score, threshold=0.8):
    if similarity_score > threshold:
        return "Same / Paraphrased"
    elif similarity_score > 0.8:
        return "Possibly Related"
    else:
        return "Different"


if __name__ == "__main__":
    # Example answers
    answer1 = "The quick brown fox jumps over the lazy dog."
    answer2 = "The quick brown fox jumps over the lazy dog."

    # Compute similarity score
    similarity_score = compute_similarity(answer1, answer2)

    # Classify the similarity
    classification = classify_similarity(similarity_score)

    # Display results
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Classification: {classification}")
    def classify_similarity(similarity_score, threshold=0.8):
        if similarity_score > threshold:
            return "Same / Paraphrased"
        elif similarity_score > 0.8:
            return "Possibly Related"
        else:
            return "Different"
