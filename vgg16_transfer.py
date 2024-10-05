import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten


# Initialize parameter.
classes = ["car", "motorcycle"]
num_classes = len(classes)
image_size = 224


image_files_data = np.load("./imagefiles.npz", allow_pickle=True)
X_train, X_test, Y_train, Y_test = (
    image_files_data["X_train"],
    image_files_data["X_test"],
    image_files_data["Y_train"],
    image_files_data["Y_test"],
)


Y_train = to_categorical(Y_train, num_classes)
Y_test = to_categorical(Y_test, num_classes)


# Normalize by dividing by 255 to scale the values between 0 and 1.
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0


# Define the model.
"""
By setting include_top=False, this model can be customized to fit specific requirements.
My own fully connected layer is added.
"""
vgg_model = VGG16(
    weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3)
)


# Fully connected layer (Flatten the output and add a Dense layer with 256 units).
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
top_model.add(Dense(256, activation="relu"))
top_model.add(Dropout(0.5))

# Output layer.
top_model.add(Dense(num_classes, activation="softmax"))


model = Model(inputs=vgg_model.input, outputs=top_model(vgg_model.output))

for layer in model.layers[:15]:
    layer.trainable = False


opt = Adam(learning_rate=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size=32, epochs=13)
score = model.evaluate(X_test, Y_test, batch_size=32)

model.save("./vgg16_transfer.h5")
