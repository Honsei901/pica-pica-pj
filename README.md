# Transfer Learning Using VGG16

## Workflow

1. Data Preparation

2. Training

- Load the data set in step 1.
- Define the model, using the pre-trained `VGG16` model as the base.
- Set Adam as the optimizer and categorical_crossentropy as the loss function.
- Train the model and evaluate its performance using metrics such as accuracy and loss.

3. Save the Model

- After training, save the model in `.h5` format.

4. Build a Simple Application for Prediction Using the Model

- Build a web application using FastAPI to implement a feature where the user can upload images and perform predictions using the trained model.

## Changes from the Initial Project

https://github.com/Honsei901/podargus-strigoides-pj

In the initial phase of this project, training was performed using only the collected data without applying transfer learning. However, since the results were not satisfactory, VGG16 was used, and the fully connected layers were added and optimized according to the current objective.

- Changed the image size from 150 to 224.
- Data normalization is performed during training rather than during data generation.

## Operation Verification

- Download the `vgg16_transfer.h5` file from the repository.<br />
  Start by clicking `download row file` at https://github.com/Honsei901/pica-pica-pj/blob/main/vgg16_transfer.h5.

- Place the downloaded .h5 file in the root directory of this repository.

- Make sure Docker is installed.

- Execute the following commands:

```bash
cd docker
docker-compose up --build
```

- Once all processes are completed, open the page at http://localhost:3000/.

- Use the "Upload File" button to select an image of a car or motorcycle.

- Click the "Estimate" button to view the results.

## Reference

- [Keras Documentation: Optimizers](https://keras.io/api/optimizers/): Official Keras documentation on optimization algorithms.
