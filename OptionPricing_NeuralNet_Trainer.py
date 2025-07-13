import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers

options_df = pd.read_csv("options_cleaned.csv") # our dataset containing X and Y

full_options_df = pd.read_csv("options_withriskfreerates_andDeltaT.csv") # loaded full data for stock tickers (to be used for splitting)

# in this model, I follow a split of 70%, 20% and 10% splits for training, dev/validation and test sets respecitvely
# instead of splitting the data fully randomly, 
# I go into the option data of each stock/ticker seperately and then perform the 70-20-10 split 
# This ensures that each of the training, dev/validation and test sets contain equal amounts of data from each stock

tickers = ["AAPL","NVDA","TSLA","AMD","AMZN","MSFT","META","SPY"]

train_list = []
dev_list = []
test_list = []

for ticker in tickers:

    df = options_df[full_options_df["stockTicker"] == ticker]

    train, dev, test = np.split(df.sample(frac=1,random_state=42), 
                       [int(0.7*len(df)), int(.9*len(df))])
    
    train_list.append(train)
    dev_list.append(dev)
    test_list.append(test)

train_df = pd.concat(train_list, ignore_index=True)
dev_df = pd.concat(dev_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

# splitting X (features) and Y (target)

target_col = "lastPrice"

Y_train = train_df[[target_col]]
X_train = train_df.drop(columns=[target_col])

Y_dev = dev_df[[target_col]]
X_dev = dev_df.drop(columns=[target_col])

Y_test = test_df[[target_col]]
X_test = test_df.drop(columns=[target_col])

# deep learning also works better when the features are normalised, so a standard scaling is done
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled = scaler.transform(X_dev)
X_test_scaled = scaler.transform(X_test)

# Define the model using the framework available on TensorFlow

model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Explicit Input layer
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

optimizer = Adam(learning_rate=0.0005)  

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_dev, Y_dev),
    epochs=100,              # Start with a small number, like 50
    batch_size=32,          # Common choice
    verbose=1               # Shows progress per epoch
)

# The above step will show the result of the training
# With this, the best model obtained has an mae of around 11
# For our dataset this is good, let's try to look if better results can be achieved by tuning the parameters

# For tuning and searching the hyperparameter space for the best model, one can use keras-tuner

def build_model(hp):
    model = keras.Sequential()

    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", 32, 256, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"])
            )
        )

    model.add(layers.Dense(1, activation='linear'))

    # Explicit mapping to avoid any naming issues
    optimizer_name = hp.Choice("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    optimizer_classes = {
        "adam": keras.optimizers.Adam,
        "rmsprop": keras.optimizers.RMSprop,
        "sgd": keras.optimizers.SGD
    }

    model.compile(
        optimizer=optimizer_classes[optimizer_name](learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"]
    )

    return model

tuner = kt.RandomSearch(
    build_model,
    objective="val_mae",
    max_trials=10,              # Try 10 different hyperparameter combinations
    executions_per_trial=1,     # How many times to train each model
    directory="kt_tuner",
    project_name="option_price_model"
)



early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

tuner.search(
    X_train, Y_train,
    validation_data=(X_dev, Y_dev),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
for param in best_hp.values:
    print(f"{param}: {best_hp.get(param)}")

# Evaluate the best model
test_loss, test_mae = best_model.evaluate(X_test, Y_test)
print("Test MAE:", test_mae)

best_model.save("best_model_from_tuner.keras") # saving the model for future use

X_test["lastPrice"] = Y_test["lastPrice"]

X_test.to_csv("test_data_options.csv",index=False) # saving the test data so we have the same test set later as well


