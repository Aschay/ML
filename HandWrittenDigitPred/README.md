# MNIST CNN Classification

A CNN-based image classification project using TensorFlow/Keras and the MNIST dataset.

## Dataset

MNIST contains 70,000 handwritten digit images:

* 54,000 training
* 6,000 validation
* 10,000 test

Images are normalized from `uint8 [0,255]` to `float32 [0,1]`.

## Data Pipeline

```text
TFDS
 ↓
Normalization
 ↓
Cache
 ↓
Shuffle (training only)
 ↓
Batch (128)
 ↓
Prefetch
```

## Model

```text
28×28×1
 ↓
Conv2D(32) → Conv2D(32)
 ↓
MaxPooling → Dropout(0.25)
 ↓
Conv2D(64) → Conv2D(64)
 ↓
MaxPooling → Dropout(0.25)
 ↓
Flatten
 ↓
Dense(128)
 ↓
Dropout(0.5)
 ↓
Dense(10)
```

The final layer outputs 10 **logits**, one for each digit (`0–9`).

Dropout is used as regularization to reduce overfitting. A`25%` dropout rate is used after the convolution blocks as a moderate level of regularization, while `50%` is used before the final classifier because the fully connected layer has substantially more parameters and can be more prone to overfitting.

## Training

* Optimizer: RMSprop
* Loss: Sparse Categorical Crossentropy (`from_logits=True`)
* Metric: Sparse Categorical Accuracy
* Batch size: 128
* Maximum epochs: 10
* Early stopping: monitors `val_loss` to prevent unnecessary training
* Patience: 3 epochs, allowing some validation-loss fluctuation before stopping
* Best weights restored after training

The model uses logits instead of applying softmax in the final layer. With `from_logits=True`, the loss function applies the required softmax operation internally. This is numerically more stable than calculating softmax probabilities separately.

## Results

| Metric                   |     Result |
| ------------------------ | ---------: |
| Best validation accuracy |     99.32% |
| Test accuracy            | **99.52%** |
| Test loss                |     0.0173 |

## Deployment
Possible approaches:

1. REST API with direct image upload
2. Edge/local inference(model compression with quantization and pruning )

For direct API uploads and object-storage uploads, client-side resizing/compression can reduce network transfer size.