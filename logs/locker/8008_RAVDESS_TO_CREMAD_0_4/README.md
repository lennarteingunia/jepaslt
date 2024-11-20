# Testing the RAVDESS models on CREMA-D

In  this directory you can find the direct output, combined outputs, and remapped outputs.

Remapping means, that the outputs of the RAVDESS models were mapped to the corresponding CREMA-D emotions.

Since Calm and Surprise do not exist in CREMA-D these were mapped to emotions 6 and 7.

#### Evaluation

To evaluate the model with "Calm" and "Neutral" combined, you need to combine columns 0 and 6 to.

Otherwise we need to drop both surprise and Calm and evaluate then. Keep in mind, to only evaluate on the first 6x6 matrix.
