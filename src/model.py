import tensorflow as tf

# Compile and fit models
def compile_and_fit(model, X_train, y_train, X_test, y_test, MAX_EPOCHS):

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = model.fit(x=X_train, y=y_train,
                        batch_size=10,
                        epochs=MAX_EPOCHS,
                        verbose=2,
                        validation_data=(X_test,y_test))

    return history

# Dense Neural Network Definition
Dense = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu', input_dim=4),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
    ])
