Here is a detailed explanation of the code for the `README.md` file:

---

# LayoutLM for Document Understanding and Information Extraction

This project demonstrates the usage of **LayoutLM**, a pre-trained transformer model for document image processing, to extract structured information from documents. The **FUNSD dataset** is used as an example, which contains documents in the form of images, text, and bounding boxes. LayoutLM utilizes both textual and layout features to enhance understanding, making it particularly useful for document classification tasks like named entity recognition (NER).

## Overview of the Code

The following steps walk through how the code processes the FUNSD dataset, performs inference using LayoutLM, and visualizes the results.

---

### Step 1: Load the FUNSD Dataset

```python
dataset = load_dataset("nielsr/funsd", trust_remote_code=True)
```

In this step, we load the **FUNSD dataset** using the `load_dataset` function from the `datasets` library. The dataset consists of document images along with their corresponding words, bounding boxes, and NER tags. This is a standard dataset used for document understanding tasks.

- `trust_remote_code=True`: Ensures that remote code from the dataset provider is trusted when loading the dataset.

---

### Step 2: Initialize the LayoutLM Tokenizer

```python
tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")
```

Here, we initialize the **LayoutLM tokenizer** using the pre-trained `microsoft/layoutlm-base-uncased` model. The tokenizer is responsible for converting input text into token IDs and mapping them to corresponding bounding box coordinates. This is essential as LayoutLM uses both text and spatial information (bounding boxes) to understand the layout of documents.

---

### Step 3: Define the Preprocessing Function

```python
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
```

This function preprocesses each example in the dataset. The preprocessing involves tokenizing the words in the document (as they are provided in a split form), padding and truncating them to a maximum length of 512 tokens.

- `padding='max_length'`: Ensures all sequences are padded to the maximum length.
- `truncation=True`: Truncates sequences to the maximum length.
- `return_offsets_mapping=True`: Returns the mapping of word offsets, useful for mapping bounding boxes.
- `is_split_into_words=True`: Assumes the input is already split into words.

---

### Step 4: Process NER Tags and Bounding Boxes

```python
labels = []
boxes = []
for i, word_id in enumerate(encoding.word_ids()):
    if word_id is None:
        labels.append(-100)
        boxes.append([0, 0, 0, 0])
    else:
        labels.append(example['ner_tags'][word_id])
        boxes.append(example['bboxes'][word_id])

encoding['labels'] = labels
encoding['bbox'] = boxes
```

For each tokenized word, the function maps the corresponding **NER tags** and **bounding box coordinates** to the tokenized words. The bounding boxes are scaled to fit the input image. The `word_ids()` function is used to determine the alignment of each token in the sequence with the original words in the document.

- If the word corresponds to a special token (e.g., padding), the bounding box is set to `[0, 0, 0, 0]` and the label is set to `-100` (which tells the model to ignore it during training).
- Otherwise, the corresponding NER tag and bounding box are added.

---

### Step 5: Load the Pre-trained LayoutLM Model

```python
model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(dataset["train"].features["ner_tags"].feature.names)
)
```

We load the pre-trained **LayoutLM model** for token classification. The model is specifically designed for tasks like named entity recognition (NER) and takes both tokenized text and bounding boxes as input.

- `num_labels`: Defines the number of unique labels (i.e., NER tags) used in the task, which is determined from the dataset.

---

### Step 6: Move Model and Inputs to the GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

The model and inputs are moved to the GPU if available, using `cuda`, otherwise, they are moved to the CPU. This ensures faster inference when a GPU is available.

---

### Step 7: Perform Inference

```python
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, bbox=bbox, labels=labels)
    logits = outputs.logits
```

Inference is performed in this step. The model processes the input IDs, attention mask, bounding boxes, and labels (for training purposes), and produces logits, which are the raw scores for each token.

- `torch.no_grad()`: Ensures that no gradients are calculated during inference, which saves memory and computation.

---

### Step 8: Decode the Predicted Labels

```python
predicted_labels = torch.argmax(logits, dim=2)
label_map = {i: label for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
predicted_labels = predicted_labels.cpu().numpy()[0]
decoded_labels = [label_map[label_id] for label_id in predicted_labels]
```

After obtaining the logits, we decode the predicted labels. The `torch.argmax` function finds the label with the highest score for each token, and these are mapped back to their human-readable label names.

---

### Step 9: Visualize Predictions on the Image

```python
image = Image.open(example["image_path"])
draw = ImageDraw.Draw(image)
```

In this step, we load the **original document image** and prepare to annotate it with predicted labels and bounding boxes. The `ImageDraw.Draw` object allows us to draw on the image.

---

### Step 10: Draw Bounding Boxes and Labels on the Image

```python
for box, label in zip(example["bboxes"], decoded_labels):
    if label != "O":
        color = colors.get(label, "blue")
        scaled_box = [
            box[0] * image_width / 1000,
            box[1] * image_height / 1000,
            box[2] * image_width / 1000,
            box[3] * image_height / 1000
        ]
        draw.rectangle(scaled_box, outline=color, width=2)
        draw.text((scaled_box[0], scaled_box[1] - 10), label, fill=color, font=font)
```

For each token in the example, we draw its corresponding bounding box on the image. The bounding boxes are scaled to the image dimensions and drawn with the predicted labels. Each label is colored according to its type, and the text is placed above the corresponding box.

---

### Step 11: Display the Annotated Image

```python
plt.figure(figsize=(12, 12))
plt.imshow(image)
plt.axis('off')
plt.show()
```

Finally, the annotated image is displayed using `matplotlib`. The bounding boxes and labels are visible on the image, allowing for a clear visual representation of the model's predictions.

---

# Output:
![Screenshot 2024-12-15 135454](https://github.com/user-attachments/assets/d01c4041-2bd7-4122-9a5e-1c173f936612)


## Conclusion

This code demonstrates the integration of text and layout features using **LayoutLM** for document understanding. By leveraging both text and bounding box information, LayoutLM is capable of effectively performing tasks like NER in documents, significantly improving the accuracy of document understanding models.

## Reference

For more information on how to use LayoutLM for document understanding, check out the [KDnuggets tutorial](https://www.kdnuggets.com/how-to-layoutlm-document-understanding-information-extraction-hugging-face-transformers).

---

This explanation should give a clear understanding of each step in the code, and how the LayoutLM model is used for document understanding and information extraction tasks. You can include this in the `README.md` of your GitHub repository for others to understand the flow of the code.
