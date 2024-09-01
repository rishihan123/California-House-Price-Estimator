import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import fetch_california_housing
import random

# Ensures reproducibility. These are set so the outliers in the data don't make a massive difference when the model is trained
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

######      CLASSES     ######

class AppException(Exception): #object for exception handling
    def __init__(self, value):
        self.__value = value
    def toString(self):
        return self.__value
    
######      FUNCTIONS      ######

def getHomeDetails():
    print("Please enter the details of the property for price prediction:")

    # Median Income
    try:
        print("What is the median income of households in your neighborhood? (in tens of thousands of dollars)")
        med_inc = float(input("> "))
        if med_inc <= 0:
            raise AppException("Median income must be a positive number.")
    except ValueError:
        raise AppException("Invalid input for median income.")
    
    # House Age
    try:
        print("What is the median age of houses in your neighborhood? (in years)")
        house_age = float(input("> "))
        if house_age < 0:
            raise AppException("House age cannot be negative.")
    except ValueError:
        raise AppException("Invalid input for house age.")
    
    # Average Number of Rooms
    try:
        print("What is the average number of rooms per house in your neighborhood?")
        ave_rooms = float(input("> "))
        if ave_rooms <= 0:
            raise AppException("Average number of rooms must be a positive number.")
    except ValueError:
        raise AppException("Invalid input for average number of rooms.")
    
    # Average Number of Bedrooms
    try:
        print("What is the average number of bedrooms per house in your neighborhood?")
        ave_bedrms = float(input("> "))
        if ave_bedrms <= 0:
            raise AppException("Average number of bedrooms must be a positive number.")
    except ValueError:
        raise AppException("Invalid input for average number of bedrooms.")
    
    # Population
    try:
        print("What is the population of your neighborhood?")
        population = float(input("> "))
        if population <= 0:
            raise AppException("Population must be a positive number.")
    except ValueError:
        raise AppException("Invalid input for population.")
    
    # Average Number of Occupants
    try:
        print("What is the average number of occupants per household?")
        ave_occup = float(input("> "))
        if ave_occup <= 0:
            raise AppException("Average number of occupants must be a positive number.")
    except ValueError:
        raise AppException("Invalid input for average number of occupants.")
    
    # Latitude
    try:
        print("Enter the latitude of your location:")
        latitude = float(input("> "))
        if not (-90 <= latitude <= 90):
            raise AppException("Latitude must be between -90 and 90.")
    except ValueError:
        raise AppException("Invalid input for latitude.")
    
    # Longitude
    try:
        print("Enter the longitude of your location:")
        longitude = float(input("> "))
        if not (-180 <= longitude <= 180):
            raise AppException("Longitude must be between -180 and 180.")
    except ValueError:
        raise AppException("Invalid input for longitude.")
    
    # Create a list of user inputs as features for the model
    user_features = [med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]
    
    return user_features

def train_and_save_model():
    data = fetch_california_housing(as_frame=True) # Load the dataset
    X = data.data #Matrix of the data
    y = data.target #List of target data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splits the dataset into training data and testing data. 0.2 means 20% of the data used for testing and 80% for training
    scaler = StandardScaler() # Initialize and fit the scaler
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'scaler.pkl') #Save the scaler to be used for the user inout data to ensure consistency
    model = tf.keras.Sequential([ #Initializes a sequential model which allows layes to be connected in a sequence
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), #ReLu allows non-linearity to be used by the model allowing for more complex training
                            kernel_regularizer=regularizers.l2(0.01)), #Regularizers help prevent overfitting by penalizing large values
        tf.keras.layers.Dropout(0.5), #Dropout layer randomly drops 50% of neurons used by the model to help prevent overfitting
        tf.keras.layers.Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1) #Output layer with 1 neuron
    ])
    model.compile(optimizer='adam', loss='mean_squared_error') #Compiles the model with the chosen optimizer and defines the mean squared error as the loss function as it is a regression problem
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) #Stops the training once the model stops improving
    history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, batch_size=32, callbacks=[early_stopping], verbose=2) #Trains the model with the training data and iterates 100 times
    model.save('california_housing_model.keras')

    # Plot training history to see if the model is actually improving or not. Lower loss values = better
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.show()

def load_model_and_scaler():
    model = tf.keras.models.load_model('california_housing_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def predict_price(user_features):
    model, scaler = load_model_and_scaler()
    user_features = np.array(user_features).reshape(1, -1) #Converts the user input into a NumPy array and reshapes it so it can be used by the model
    user_features_scaled = scaler.transform(user_features) #Scales the user input with the same scaler that was used to scale the training data
    predicted_price = model.predict(user_features_scaled) #The user input is passed into the model to predict a house price based on the input and the training data
    return predicted_price[0][0] * 100000

######      APP      ######

#train_and_save_model() #This is used to train the model.

try:
    user_features = getHomeDetails()
except AppException as e:
    print(e.toString())
predicted_price = predict_price(user_features)
print(f"Predicted house price: ${predicted_price:,.2f}")
