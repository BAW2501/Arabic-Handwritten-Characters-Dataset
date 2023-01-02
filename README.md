# Arabic Handwritten Characters Dataset
This dataset contains images of handwritten Arabic characters, with a total of 16,800 images. Each image is 32x32 pixels in size and is grayscale. The characters are drawn from 28 different classes, representing each of the letters in the Arabic alphabet.

## Model Description
To recognize the handwritten characters in this dataset, we used a Convolutional Neural Network (CNN) model. CNNs are well-suited for image recognition tasks because they can learn features from the data and build hierarchical representations of the input.

The model we used for this task consists of two convolutional layers, followed by two max pooling layers. The output of the pooling layers is then flattened and passed through two fully connected (dense) layers, with the final layer containing 28 units and using a softmax activation function to produce the final output class predictions. The model uses the Adam optimizer and categorical cross-entropy loss, and is trained for 10 epochs with a batch size of 32.

## Here is a summary of the model architecture:

```
INPUT → CONV → RELU → POOL → CONV → RELU → POOL → FC → RELU → FC
```
* The first convolutional layer has 32 filters with a kernel size of (5,5) and uses a ReLU activation function. The output image size is ((32-5)+1) = 28.
* The first max pooling layer has non-overlapping regions and down-samples by a factor of 2 in each direction, resulting in an output image size of 28/2 = 14.
* The second convolutional layer has 64 filters with a kernel size of (5,5) and uses a ReLU activation function. The output image size is ((14-5)+1) = 10.
* The second max pooling layer has non-overlapping regions and down-samples by a factor of 2 in each direction, resulting in an output image size of 10/2 = 5.

* The output of the max pooling layer is flattened, resulting in an input size of 5x5x128=1600 for the first fully connected (dense) layer. This layer has 1024 units and uses a ReLU activation function.
* The final fully connected layer has 28 units and uses a softmax activation function to produce the final output class predictions.

After training, the model is evaluated on a separate test set to measure its performance. The model's accuracy is calculated by comparing the predicted classes to the true classes in the test set. In addition, a confusion matrix and classification report can be generated to further analyze the model's performance.

## Model Performance
Overall, this CNN model is effective at recognizing handwritten Arabic characters, achieving high accuracy on the dataset. However, there may be ways to further improve the model's performance, such as by fine-tuning the model's hyperparameters or incorporating additional data augmentation techniques.

here is the classification report

              precision    recall  f1-score   support

           أ       0.98      1.00      0.99       120
           ب       0.95      0.97      0.96       120
           ت       0.82      0.97      0.89       120
           ث       0.96      0.88      0.92       120
           ج       0.95      0.95      0.95       120
           ح       0.88      0.96      0.92       120
           خ       0.90      0.88      0.89       120
           د       0.87      0.93      0.90       120
           ذ       0.84      0.92      0.88       120
           ر       0.85      0.93      0.89       120
           ز       0.91      0.82      0.86       120
           س       0.98      0.93      0.95       120
           ش       0.97      0.95      0.96       120
           ص       0.93      0.97      0.95       120
           ض       0.99      0.85      0.91       120
           ط       0.93      0.93      0.93       120
           ظ       0.95      0.88      0.91       120
           ع       0.88      0.90      0.89       120
           غ       0.95      0.88      0.92       120
           ف       0.93      0.82      0.87       120
           ق       0.82      0.94      0.88       120
           ك       0.97      0.94      0.95       120
           ل       0.99      0.98      0.99       120
           م       0.97      0.97      0.97       120
           ن       0.95      0.83      0.89       120
           ه       0.98      0.78      0.87       120
           و       0.79      0.97      0.87       120
           ي       0.93      0.96      0.94       120

      accuracy                              0.92      3360
      macro avg         0.92      0.92      0.92      3360
      weighted avg      0.92      0.92      0.92      3360

# References

* Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, 1998.
* A. Krizhevsky, I. Sutskever, and G. E. Hinton. "Imagenet classification with deep convolutional neural networks." In Advances in neural information processing systems, pp. 1097-1105, 2012.
* C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. "Going deeper with convolutions." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1-9, 2015.
* W. H. Al-Hamad, M. F. Al-Rousan, and A. K. A. Al-Widyan. "Arabic handwritten character recognition using convolutional neural network." WSEAS Transactions on Computer Research, 5(1):11-19, 2017.