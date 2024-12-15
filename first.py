from datasets import load_dataset
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Step 1: Load the FUNSD dataset
# The dataset contains documents in the form of images, text, and bounding boxes
dataset = load_dataset("nielsr/funsd", trust_remote_code=True)

# Step 2: Initialize the LayoutLM tokenizer
# The tokenizer will convert words into token IDs and handle bounding box inputs
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

# Step 3: Define a function to preprocess the examples
# This function tokenizes the input and maps NER tags and bounding boxes
def preprocess_example(example):
    # Tokenize the input example
    encoding = tokenizer(
        example['words'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        is_split_into_words=True
    )
    
    # Initialize lists for labels and bounding boxes
    labels = []
    boxes = []
    
    # Use the method properly: call `word_ids()`
    for i, word_id in enumerate(encoding.word_ids()):  # Correct usage with parentheses
        if word_id is None:  # If it's a special token (e.g., [PAD]), ignore it
            labels.append(-100)  # Use -100 to ignore this during loss calculation
            boxes.append([0, 0, 0, 0])  # Dummy bounding box
        else:
            labels.append(example['ner_tags'][word_id])  # Map the NER tag
            boxes.append(example['bboxes'][word_id])    # Map the bounding box

    # Add labels and bounding boxes to the encoding
    encoding['labels'] = labels
    encoding['bbox'] = boxes  # Use 'bbox' to match LayoutLM's input naming convention
    return encoding


# Step 4: Preprocess one example for testing
# Preprocess the first example in the training dataset
example = dataset["train"][0]  # Select one example
encoding = preprocess_example(example)

# Step 5: Load the pre-trained LayoutLM model
# This model is designed for token classification tasks with bounding box input
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(dataset["train"].features["ner_tags"].feature.names)  # Number of unique labels
)

# Step 6: Move the model and inputs to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Convert inputs to PyTorch tensors and move to the device
input_ids = torch.tensor(encoding["input_ids"]).unsqueeze(0).to(device)
attention_mask = torch.tensor(encoding["attention_mask"]).unsqueeze(0).to(device)
bbox = torch.tensor(encoding["bbox"]).unsqueeze(0).to(device)
labels = torch.tensor(encoding["labels"]).unsqueeze(0).to(device)

# Step 7: Perform inference with the model
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, labels=labels)
    logits = outputs.logits  # Get the predicted logits

# Step 8: Decode the predicted labels
# Map predicted token IDs to their corresponding label names
predicted_labels = torch.argmax(logits, dim=2)  # Get the index of the highest score
label_map = {i: label for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
predicted_labels = predicted_labels.cpu().numpy()[0]  # Move to CPU and extract data
decoded_labels = [label_map[label_id] for label_id in predicted_labels]

# Step 9: Visualize predictions on the image
# Load the original document image
image = Image.open(example["image_path"])

# Create a drawing context for the image
draw = ImageDraw.Draw(image)

# Define colors for each label
colors = {
    "I-HEADER": "blue",
    "I-QUESTION": "green",
    "I-ANSWER": "red",
    "B-HEADER": "yellow",
    "B-QUESTION": "purple",
    "B-ANSWER": "orange",
    "O": "white"  # Outside (no label)
}

# Get image dimensions
image_width, image_height = image.size

# Define a font for label text (fallback to default if custom font is unavailable)
try:
    font = ImageFont.truetype("arial.ttf", size=12)
except:
    font = ImageFont.load_default()

# Draw bounding boxes and labels on the image
for box, label in zip(example["bboxes"], decoded_labels):
    if label != "O":  # Ignore tokens with the "Outside" label
        color = colors.get(label, "blue")  # Get color for the label
        # Scale bounding box coordinates to image dimensions
        scaled_box = [
            box[0] * image_width / 1000,
            box[1] * image_height / 1000,
            box[2] * image_width / 1000,
            box[3] * image_height / 1000
        ]
        # Draw the bounding box
        draw.rectangle(scaled_box, outline=color, width=2)
        # Add the label text above the bounding box
        draw.text((scaled_box[0], scaled_box[1] - 10), label, fill=color, font=font)

# Step 10: Display the annotated image
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')  # Remove axes for better visualization
plt.show()
