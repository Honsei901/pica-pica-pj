import glob
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split


# Initialize parameter.
classes = ["car", "motorcycle"]
num_classes = len(classes)
image_size = 224

X = []
Y = []

# Load images.
for index, classlabel in enumerate(classes):
    photos_dir = "./assets" "/" + classlabel
    files = glob.glob(photos_dir + "/" + "*.jpg")

    for file in files:
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.array(image)

        X.append(data)
        Y.append(index)

# Convert the images to NumPy arrays.
X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
np.savez(
    "./imagefiles.npz", X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test
)
