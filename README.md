# Brain Tumor Detection from MRI Images

This project classifies brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. It combines a deep-learning training notebook with a small Flask web interface for trying a trained model on an uploaded image.

> **Educational project only.** This model is not a medical device and its output is not a diagnosis. Do not use it for clinical decisions.

## How the project works

The workflow has two parts:

1. **Train and evaluate** the image classifier in `brain_tumour_detection_using_deep_learning.ipynb`.
2. **Load the saved model** in `main.py`, upload an MRI image in the browser, and display the predicted class and its score.

The repository includes neither the full MRI dataset nor the trained model file. You must supply both (the model is needed only to run the web app).

## Techniques used

### Transfer learning with VGG16

The notebook uses **VGG16**, a convolutional neural network originally trained on ImageNet, as a feature extractor. Its original classification head is removed (`include_top=False`) and its pretrained weights are loaded. Most VGG16 layers are frozen so their learned visual features stay fixed; the final three base-model layers are unfrozen for fine-tuning on MRI images.

A new classification head is added:

- `Flatten` turns the VGG16 feature maps into a vector.
- `Dropout(0.3)` and `Dropout(0.2)` randomly drop activations during training to help reduce overfitting.
- A 128-unit fully connected layer with ReLU activation learns MRI-specific combinations of features.
- A four-unit softmax layer returns a score for each class.

The model is compiled with the **Adam optimizer** (learning rate `0.0001`) and **sparse categorical cross-entropy** loss. Sparse cross-entropy allows the target classes to be supplied as integer IDs rather than one-hot vectors. The notebook trains for five epochs with a batch size of 20.

### Image preparation and augmentation

Images are resized to **128 × 128** pixels and treated as RGB. Pixel values are divided by 255 so they fall in `[0, 1]`. During training, the notebook randomly adjusts brightness and contrast within a range of 0.8 to 1.2. This augmentation exposes the model to modest image variation. It is applied by the notebook's image-loading helper; the test-loading cell also calls that helper, so the test images receive random brightness/contrast changes as written.

The notebook gathers image paths from class-named directories, shuffles the paths and labels, then encodes class names as integer labels. It uses a generator to load images in batches instead of loading the entire training set at once.

### Evaluation

The notebook predicts labels for the test images and includes:

- a **classification report** (precision, recall, and F1 score by class),
- a **confusion matrix** to show which classes are confused with one another,
- per-class **ROC curves and AUC** plots, and
- plots of training accuracy and loss.

These are evaluation tools, not evidence of clinical performance. The repository does not include dataset details, saved evaluation outputs, or results that would establish how well the model generalizes.

## Prediction flow in the web app

When a user submits an image, the Flask route in `main.py` saves it under `uploads/`, resizes it to 128 × 128, scales pixel values to `[0, 1]`, and passes it to the loaded Keras model. The model returns four class scores. The app selects the highest-scoring class with `argmax` and displays that class alongside its maximum score as a percentage. `notumor` is shown as **No Tumor**; the other labels are shown as **Tumor: <class>**.

The displayed percentage is the model's softmax score. It should not be interpreted as a medically calibrated probability or certainty.

## Project files

| File | Purpose |
| --- | --- |
| `brain_tumour_detection_using_deep_learning.ipynb` | Colab-oriented data loading, augmentation, VGG16 training, evaluation, model saving, and example predictions. |
| `main.py` | Flask server and image prediction logic. |
| `index.html` | Bootstrap-styled upload form and result page rendered by Flask. |
| `requirements.txt` | Pinned Python packages for the app and much of the notebook workflow. |
| `Te-gl_0015.jpg` | Example test image with `glioma` label in its filename. |
| `Te-meTr_0001.jpg` | Example test image with `meningioma` label in its filename. |
| `Te-noTr_0004.jpg` | Example test image with `no tumor` label in its filename. |
| `Te-piTr_0003.jpg` | Example test image with `pituitary` label in its filename. |

## Dataset layout

The notebook expects separate `Training` and `Testing` folders, each containing one subfolder per class:

```text
MRI Images/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The notebook currently mounts Google Drive in Colab and uses these paths:

```text
/content/drive/MyDrive/MRI Images/Training/
/content/drive/MyDrive/MRI Images/Testing/
```

Change the paths if your data lives elsewhere. Keep the class folder names consistent with the labels expected by the app.

## Train and save the model

1. Open `brain_tumour_detection_using_deep_learning.ipynb` in Google Colab and provide the dataset in the layout above.
2. Update `train_dir` and `test_dir` if needed, then run the notebook cells in order.
3. The notebook's save cell writes `model.h5` to its current working directory.
4. In the project root, create a `models` directory and copy the file there as `models/model.h5`.

Training uses ImageNet weights for VGG16, which may need to be downloaded the first time. A GPU can speed up training but is not required by the code.

## Install and run the Flask app

Use a Python environment compatible with the pinned dependencies (including TensorFlow 2.18.0 and Keras 3.7.0). From the project root:

```bash
python -m venv .venv
```

Activate the environment in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure `models/model.h5` exists, then launch the server:

```bash
python main.py
```

Visit <http://127.0.0.1:5000/>, choose an MRI image, and submit it to see the predicted label and score. Uploaded files are stored in `uploads/`. The app runs with Flask debug mode enabled, so use it for local development only.

## Important implementation details

- **Class order must match.** `main.py` assumes model outputs are ordered as `pituitary`, `glioma`, `notumor`, `meningioma`. The notebook encodes labels using the order returned by the filesystem when listing training folders, which may differ. Confirm the training index-to-class mapping and update one side before relying on predictions.
- **Notebook dependencies:** the notebook uses scikit-learn, Matplotlib, and Seaborn, which are not listed in `requirements.txt`. For local notebook evaluation, install them with `python -m pip install scikit-learn matplotlib seaborn` if needed.
- **Input handling:** the web form accepts image files, but the current Flask route does not validate file types or impose upload limits. Use only trusted local inputs.
- **Model and data availability:** neither `models/model.h5` nor the full dataset is checked into this repository.

## License and data attribution

The repository does not include a license or a dataset citation. Add the appropriate license and cite the dataset source and terms before redistributing the code, model, or data.
