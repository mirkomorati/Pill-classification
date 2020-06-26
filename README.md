# Pill-classification

## Dataset

- ftp://ftp.nlm.nih.gov/nlmdata/pir/DC.zip
- https://pir.nlm.nih.gov/challenge/submission.html
- Utility to download images based on size
- Decide how to divide the data (learn set, test set)

### Bonus

- Batch pills classification: need to create a new dataset

## Pipeline

- Segmentation
  - Binarization
  - Edge detection (Sobel, Canny)
- Normalization
  - Centroid
- Feature extraction
  - Color
  - Shape
  - Texture
  - Text
- Classification
  - SVM
  - NN
  - Naive Bayes

