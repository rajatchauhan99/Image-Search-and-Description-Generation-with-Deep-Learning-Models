# Image-Search-and-Description-Generation-with-Deep-Learning-Models


## Project Description: Image Search and Description Generation with Deep Learning Models

### Overview:
This project aims to demonstrate an end-to-end system for searching similar images based on textual queries and generating descriptions for images using state-of-the-art deep learning models. The system utilizes pre-trained models for feature extraction from images and text, enabling users to explore image similarities through textual descriptions and generate textual descriptions for input images.

### Features:
1. **Image Feature Extraction**:
   - Utilizes a pre-trained ResNet model to extract features from images.
   - Removes the fully connected layer to obtain high-level image representations.

2. **Image Vector Database Creation**:
   - Constructs a vector database containing image features for efficient search.

3. **Text Feature Extraction**:
   - Incorporates a pre-trained BERT model to extract features from input textual queries.
   - Implements mean pooling of BERT outputs for text feature representation.

4. **Dimensionality Reduction**:
   - Uses Principal Component Analysis (PCA) to reduce image vector dimensionality to match BERT feature dimensions.

5. **Image Search with Text**:
   - Searches for similar images based on input textual queries.
   - Calculates cosine similarity between text features and image vectors.

6. **Description Generation**:
   - Utilizes a pre-trained GPT-2 model to generate descriptions for input images.
   - Generates descriptive text based on the provided query.

### Libraries and Models:
- **Libraries**:
  - `torch`, `torch.nn`, `torchvision`, `PIL`, `numpy`, `sklearn`, `datasets`, `transformers`, `matplotlib`

- **Pre-trained Models**:
  - **Image Processing**:
    - ResNet-50 for image feature extraction.
  - **Text Processing**:
    - BERT for text feature extraction.
    - GPT-2 for text description generation.

### Workflow:
1. **Image Preprocessing**:
   - Resizes images to 224x224 pixels.
   - Normalizes images using specified mean and standard deviation.

2. **Vector Database Creation**:
   - Iterates through a dataset (Pascal VOC 2012 in this example).
   - Extracts image features using the ResNet-50 model.
   - Stores image features in a vector database.

3. **Text Feature Extraction**:
   - Processes textual queries with the BERT tokenizer.
   - Utilizes BERT model to extract features from text.
   - Creates a representation of input text for similarity calculation.

4. **Dimensionality Reduction**:
   - Applies PCA to reduce image vector dimensionality.

5. **Search Similar Images with Text**:
   - Takes a textual query as input.
   - Calculates cosine similarity between query text features and image vectors.
   - Retrieves top-k similar images and their similarity scores.

6. **Description Generation**:
   - Generates textual descriptions for input queries using GPT-2.
   - Produces descriptive sentences based on the provided query.

### Example Usage:
```python
# Search for similar images based on a text query
query_text = "Eiffel Tower"
similar_images = search_similar_images_with_text(query_text, top_k=5)

# Display similar images with their similarity scores
fig, axs = plt.subplots(1, len(similar_images), figsize=(15, 5))
fig.suptitle("Similar Images")

for i, (img, similarity) in enumerate(similar_images):
    ax = axs[i]
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"Similarity: {similarity:.4f}")

plt.tight_layout()
plt.show()

# Generate a description for an input query
query_text = "The Eiffel Tower"
description = generate_description(query_text)
print("Generated Description:")
print(description)
```

### Conclusion:
This project provides a comprehensive example of using deep learning models for image search and description generation. By combining image and text processing techniques, users can search for visually similar images based on textual queries and generate descriptive sentences for input images. The system demonstrates the power of pre-trained models and their applications in multimedia retrieval and natural language understanding.
