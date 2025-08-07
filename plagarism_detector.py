import torch
import joblib
from transformers import BertTokenizer, BertModel

# Use CPU to avoid GPU issues
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer & model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()  # Set to eval mode


# Function to extract BERT embeddings safely
def get_bert_embeddings(text):
    # Tokenize with truncation (BERT max length is 512)
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Move tensors to the same device as the model
    tokens = {key: val.to(device) for key, val in tokens.items()}

    with torch.no_grad():  # Disable gradient computation
        outputs = bert_model(**tokens)

    # Extract [CLS] token embedding (first token representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

    return cls_embedding


if __name__ == "__main__":
    # Load trained model
    try:
        logreg_bert = joblib.load('bert_ai_text_detector.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # Sample text for classification
    text = """Key Points:
Dynamic Grid and Shelves:

Shelves are added dynamically using the add_shelves() function, and the grid layout is updated accordingly.
The 1 in the grid marks shelves as blocked locations.
90-Degree Turns:

The A* algorithm avoids diagonal movement and ensures the path includes only horizontal and vertical movements, which results in right-angle turns (90 degrees).
Attractive Visualization:

Shelves are shown as red blocks.
Order locations are shown as green circles.
The worker’s path is shown in orange, and it moves step-by-step with 90-degree turns.
Path Animation:

The animate_path() function visually animates the worker’s movement from one point to another.
It clears the previous frame to simulate the worker’s movement from one position to the next.
Smooth Transition:

The worker’s movement between positions is animated with a small pause between each step, allowing you to observe the movement and turns.
Running the Code:
The code first creates the grid, adds shelves, and sets order points.
It calculates the optimal path from the starting point to each order, then animates the pathfinding.
The worker moves in a step-by-step fashion, with each turn restricted to 90 degrees.
This approach will make the pathfinding process visually engaging, showing each step of the movement while respecting the constraints of shelf positions.

Let me know if you need further adjustments or improvements!"""


    embedding = get_bert_embeddings(text)
    embedding = embedding.reshape(1, -1)


    try:
        classes = ['human', 'ai']
        prediction = logreg_bert.predict(embedding)[0]
        print(f"Prediction: {classes[prediction]}")
    except Exception as e:
        print(f"Prediction error: {e}")