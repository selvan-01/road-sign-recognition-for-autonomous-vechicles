# =========================
# IMPORT LIBRARIES
# =========================
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


# =========================
# DATASET PREPARATION
# =========================

data = []      # Stores image data
labels = []    # Stores corresponding labels
classes = 43   # Total number of traffic sign classes

cur_path = os.getcwd()  # Get current working directory


# Dictionary for class labels
classs = {
    0:"Speed limit (20km/h)", 1:"Speed limit (30km/h)", 2:"Speed limit (50km/h)",
    3:"Speed limit (60km/h)", 4:"Speed limit (70km/h)", 5:"Speed limit (80km/h)",
    6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)", 8:"Speed limit (120km/h)",
    9:"No passing", 10:"No passing veh over 3.5 tons", 11:"Right-of-way at intersection",
    12:"Priority road", 13:"Yield", 14:"Stop", 15:"No vehicles",
    16:"Veh > 3.5 tons prohibited", 17:"No entry", 18:"General caution",
    19:"Dangerous curve left", 20:"Dangerous curve right", 21:"Double curve",
    22:"Bumpy road", 23:"Slippery road", 24:"Road narrows on the right",
    25:"Road work", 26:"Traffic signals", 27:"Pedestrians",
    28:"Children crossing", 29:"Bicycles crossing", 30:"Beware of ice/snow",
    31:"Wild animals crossing", 32:"End speed + passing limits",
    33:"Turn right ahead", 34:"Turn left ahead", 35:"Ahead only",
    36:"Go straight or right", 37:"Go straight or left",
    38:"Keep right", 39:"Keep left", 40:"Roundabout mandatory",
    41:"End of no passing", 42:"End no passing veh > 3.5 tons"
}


# =========================
# LOAD DATASET
# =========================

print("📂 Loading Images and Labels...")

for i in range(classes):
    path = os.path.join(cur_path, 'dataset/train/', str(i))
    images = os.listdir(path)

    for img_name in images:
        try:
            # Open image
            image = Image.open(os.path.join(path, img_name))

            # Resize to 30x30 (standard input size)
            image = image.resize((30, 30))

            # Convert to numpy array
            image = np.array(image)

            # Append to dataset
            data.append(image)
            labels.append(i)

        except:
            print(f"❌ Error loading image: {img_name}")

print("✅ Dataset Loaded Successfully")


# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)


# =========================
# TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)


# =========================
# UI DESIGN (PYQT5)
# =========================

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)

        # Browse Button
        self.BrowseImage = QtWidgets.QPushButton(self.centralwidget)
        self.BrowseImage.setGeometry(QtCore.QRect(160, 370, 151, 51))
        self.BrowseImage.setText("Browse Image")

        # Image Display Label
        self.imageLbl = QtWidgets.QLabel(self.centralwidget)
        self.imageLbl.setGeometry(QtCore.QRect(200, 80, 361, 261))
        self.imageLbl.setFrameShape(QtWidgets.QFrame.Box)

        # Title
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 621, 20))
        self.label_2.setText("ROAD SIGN RECOGNITION")

        # Classify Button
        self.Classify = QtWidgets.QPushButton(self.centralwidget)
        self.Classify.setGeometry(QtCore.QRect(160, 450, 151, 51))
        self.Classify.setText("Classify")

        # Training Button
        self.Training = QtWidgets.QPushButton(self.centralwidget)
        self.Training.setGeometry(QtCore.QRect(400, 450, 151, 51))
        self.Training.setText("Training")

        # Output Text Box
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(400, 390, 211, 51))

        MainWindow.setCentralWidget(self.centralwidget)

        # Button Actions
        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)


    # =========================
    # LOAD IMAGE FUNCTION
    # =========================
    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if fileName:
            self.file = fileName

            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(
                self.imageLbl.width(),
                self.imageLbl.height(),
                QtCore.Qt.KeepAspectRatio
            )

            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)


    # =========================
    # CLASSIFICATION FUNCTION
    # =========================
    def classifyFunction(self):
        model = load_model('my_model.h5')

        # Load and preprocess image
        test_image = Image.open(self.file)
        test_image = test_image.resize((30, 30))
        test_image = np.expand_dims(test_image, axis=0)
        test_image = np.array(test_image)

        # Predict
        result = model.predict(test_image)[0]
        predicted_class = result.argmax()

        sign = classs[predicted_class]

        # Display result
        self.textEdit.setText(sign)


    # =========================
    # TRAINING FUNCTION
    # =========================
    def trainingFunction(self):
        self.textEdit.setText("⏳ Training in progress...")

        # CNN Model
        model = Sequential()

        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=X_train.shape[1:]))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))

        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=5,
            validation_data=(X_test, y_test)
        )

        # Save model
        model.save("my_model_new.h5")

        # Plot Accuracy
        plt.figure()
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.savefig('accuracy.png')

        # Plot Loss
        plt.figure()
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.savefig('loss.png')

        self.textEdit.setText("✅ Model trained & saved!")


# =========================
# MAIN FUNCTION
# =========================
if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())