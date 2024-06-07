import pandas as pd
import numpy as np
import os
from keras.callbacks import Callback
from argparse import ArgumentParser
from argparse import ArgumentParser, Namespace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from gensim.models import Word2Vec
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, concatenate, BatchNormalization
from keras.layers import Layer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from keras.layers import Lambda

def extract_text_features(data, model):
    features = []
    vector_size = model.vector_size 

    for post in data['post']:
        post_vector = []
        for word in post.split():
            if word in model.wv:
                post_vector.append(model.wv[word])
        if len(post_vector) > 0:
            features.append(np.mean(post_vector, axis=0))
        else:
            features.append(np.zeros(vector_size))
    
    return np.array(features)

def data_formation(df):
    count = 0
    posts = []
    names = []
    descriptions = []
    protecteds = []
    followers_counts = []
    friends_counts = []
    listed_counts = []
    favourites_counts = []
    verifieds = []
    statuses_counts = []
    labels = []

    for row_itr in df.iterrows():
        row = row_itr[1]
        if row['tweet'] is None:
            row['tweet'] = ['']  
        for post in row['tweet']:
            posts.append(post)  
            # descriptions.append(row['profile']['description'].rstrip())
            protecteds.append(int(row['profile']['protected'] == 'True '))
            followers_counts.append(row['profile']['followers_count'].rstrip())
            friends_counts.append(row['profile']['friends_count'].rstrip())
            listed_counts.append(row['profile']['listed_count'].rstrip())
            favourites_counts.append(row['profile']['favourites_count'.rstrip()])
            verifieds.append(int(row['profile']['verified'] == 'True '))
            statuses_counts.append(row['profile']['statuses_count'].rstrip())
            labels.append(row['label'])
    d = {
        'post' : posts,
        # 'description' : descriptions,
        'protected' : protecteds,
        'followers_count' : followers_counts,
        'listed_count' : listed_counts,
        'favourites_count' : favourites_counts,
        'verified' : verifieds,
        'statuses_count' : statuses_counts,
        'label' : labels,
    }
    return pd.DataFrame(d)

def get_input_data(data, word2vec_model, include_numerical=True):
    input_features = extract_text_features(data, word2vec_model)
    if include_numerical:
        scaler = StandardScaler()
        numerical_data = data.drop(columns=['post', 'label'])
        numerical_scaled = scaler.fit_transform(numerical_data)
        input_features = np.hstack((input_features, numerical_scaled))
    return input_features

def main(args):
    # Load training data
    twibot_train_df = pd.read_json(args.twibot_train_file)
    # Load testing data
    twibot_test_df = pd.read_json(args.twibot_test_file)
    # twibot_train_df = twibot_train_df[:100]
    # twibot_test_df = twibot_test_df[:100]
    # Process data independently
    train_df = data_formation(twibot_train_df)
    test_df = data_formation(twibot_test_df)
    # train_df = train_df.sample(frac=0.55).reset_index(drop=True)
    
    # Combine for Word2Vec training
    df = pd.concat([train_df, test_df])

    if os.path.isfile("word2vec.model"):
        print('Loading Word2Vec ...')
        word2vec_model = Word2Vec.load("word2vec.model")
    else:
        print('Training Word2Vec ...')
        # Train Word2Vec for str->vec
        sentences = [post.split() for post in df['post']]
        word2vec_model = Word2Vec(sentences, vector_size=64, window=5, min_count=1, workers=8)
        word2vec_model.save("word2vec.model")
        
    print('Getting features ...')
    train_features = extract_text_features(train_df, word2vec_model)
    test_features = extract_text_features(test_df, word2vec_model)
    
    # Reshape features for CNN input
    train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], 1)
    test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], 1)
    
    # Split test data into validation and test sets
    test_features, val_features, test_df, val_df = train_test_split(test_features, test_df, test_size=0.5, random_state=42)
    
    # Define inputs
    input_word2vec = Input(shape=(train_features.shape[1], train_features.shape[2]))  # Assuming Word2Vec output shape
    scaler = StandardScaler()
    train_numerical = train_df.drop(columns=['post', 'label'])
    train_numerical_scaled = scaler.fit_transform(train_numerical)
    input_numerical = Input(shape=(train_numerical_scaled.shape[1],)) 
    val_numerical = val_df.drop(columns=['post', 'label'])
    val_numerical_scaled = scaler.transform(val_numerical)
    test_numerical = test_df.drop(columns=['post', 'label'])
    test_numerical_scaled = scaler.transform(test_numerical)
    
    # Generate mask to randomly disable numerical features
    mask = Lambda(lambda x: tf.cast(tf.random.uniform(tf.shape(x)) < 0.85, dtype=tf.float32))(input_numerical)
    masked_numerical = Lambda(lambda x: x[0] * x[1])([input_numerical, mask])

    # Word2Vec branch with CONV1D kernel size 3
    cnn_word2vec_3_1 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(input_word2vec)
    cnn_word2vec_3_2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(cnn_word2vec_3_1)
    maxpool_word2vec_3 = MaxPooling1D(pool_size=3)(cnn_word2vec_3_2)
    flatten_word2vec_3 = Flatten()(maxpool_word2vec_3)

    # Word2Vec branch with CONV1D kernel size 5
    cnn_word2vec_5_1 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(input_word2vec)
    cnn_word2vec_5_2 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(cnn_word2vec_5_1)
    maxpool_word2vec_5 = MaxPooling1D(pool_size=3)(cnn_word2vec_5_2)
    flatten_word2vec_5 = Flatten()(maxpool_word2vec_5)

    # Merge Word2Vec branches and masked_numerical branchs
    merged = concatenate([flatten_word2vec_3, flatten_word2vec_5, masked_numerical])

    # Fully connected layers with BatchNormalization and Dropout
    fcnn = Dense(64, activation='relu')(merged)
    fcnn = BatchNormalization()(fcnn)
    fcnn = Dropout(0.5)(fcnn)
    fcnn = Dense(64, activation='relu')(fcnn)
    fcnn = BatchNormalization()(fcnn)
    fcnn = Dropout(0.5)(fcnn)
    output = Dense(1, activation='sigmoid')(fcnn)

    # Define model
    model = Model(inputs=[input_word2vec, masked_numerical], outputs=output)
    model.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit([train_features, train_numerical_scaled], train_df['label'].values, epochs=20, batch_size=512, 
                        validation_data=([val_features, val_numerical_scaled], val_df['label'].values), callbacks=[early_stopping])

    # Evaluate model on test set
    loss, accuracy = model.evaluate([test_features, test_numerical_scaled], test_df['label'].values)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    y_pred = (model.predict([test_features, test_numerical_scaled]) > 0.5).astype(int)
    print(classification_report(test_df['label'], y_pred))

    # Plot training & validation loss curves
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--twibot_train_file", type=str, default='Twibot-20/train.json', help='TwiBot training file path')
    parser.add_argument("--twibot_test_file", type=str, default='Twibot-20/test.json', help='TwiBot testing file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print('GPU: ', tf.config.list_physical_devices('GPU'))
    args = parse_args()
    main(args)

