nn_features = []
nn_labels = []

for train_token, train_compelexity in train_df[['token', 'complexity']].itertuples(index=False):
    if type(train_token) is not str:
        continue
    train_token = train_token.lower()
    nn_features.append(fasttext_model.wv.get_vector(train_token))
    nn_labels.append(train_complexity)




import tensorflow as tf
import numpy as np


nn_features = np.asarray(nn_features)
nn_labels = np.asarray(nn_labels)

lr = tf.keras.Sequential([
    #tf.keras.layers.Dense(units=3500, activation=tf.keras.activations.relu),
    #tf.keras.layers.Dense(units=1500, activation=tf.keras.activations.relu),
    #tf.keras.layers.Dense(units=500, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=50, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(units=1, activation=tf.keras.activations.sigmoid)
])

lr.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MSE
)

history = lr.fit(nn_features, nn_labels, epochs=10, verbose=True)
print("Average test loss: ", history.history['loss'][-1])






for df in [train_df, test_df]:
    predictions = []
    train_complexities = []
    for train_token, train_complexity in df[['token', 'complexity']].itertuples(index=False):
        if type(train_token) is not str:
            continue
        train_token = train_token.lower()
        vec = np.asarray([fasttext_model.wv.get_vector(train_token)])
        prediction = lr.predict(vec)
        predictions.append(prediction)
        train_complexities.append(train_compelexity)
    print('MSE=', float(np.mean(tf.keras.losses.MSE(train_complexities, predictions))))