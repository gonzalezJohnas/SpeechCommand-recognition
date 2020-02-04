import argparse

# Data Loading
import pickle

from tensorflow.keras.backend import squeeze
from utils import *
import models

random.seed(42)

if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))

    print("Eager execution: {}".format(tf.executing_eagerly()))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--indir',
        help='Aboslute path to data directory containing .wav files',
        required=True
    )

    parser.add_argument(
        '--serialize',
        help='Loading from serialize object',
        required=False,
        default=False
    )

    parser.add_argument(
        '--checkpoint_path',
        help='Checkpoint path of a previous weight model',
        required=False,
        default=None
    )

    parser.add_argument(
        '--model',
        help='Model path of a previous trained model',
        required=False,
        default=None
    )

    parser.add_argument(
        '--model-type',
        help='Model to use possible value : {cnn, lstm, attention_lstm}',
        required=False,
        default=None
    )

    args = parser.parse_args()

    if args.serialize:
        trainset = pickle.load(open(os.path.join(args.indir, 'trainset.p'), 'rb'))
        valset = pickle.load(open(os.path.join(args.indir, 'valset.p'), 'rb'))
        testset = pickle.load(open(os.path.join(args.indir, 'testset.p'), 'rb'))

    else:

        print("Loading wave file")
        trainset, valset, testset = load_data(args.indir)

        pickle.dump(trainset, open("data/trainset.p", "wb"))
        pickle.dump(valset, open("data/valset.p", "wb"))
        pickle.dump(testset, open("data/testset.p", "wb"))

    feature_shape = np.expand_dims( trainset[0][2], -1).shape
    print(
        "The dataset is divide with: \n - {} training samples \n - {} validation samples \n - {} testing samples \n \
         Sample shape {} with {} labels".format(
            len(trainset), len(valset),
            len(testset), feature_shape, len(LABELS)))

    print("Creating Tensorflow dataset")

    dataset_train = tf.data.Dataset.from_tensor_slices(format_dataset(trainset)).shuffle(buffer_size=100).batch(
        BATCH_SIZE)
    dataset_validation = tf.data.Dataset.from_tensor_slices(format_dataset(valset)).shuffle(buffer_size=100).batch(
        BATCH_SIZE)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-4,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=15,
            verbose=1), tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
    ]

    if args.model:
        model = tf.keras.models.load_model(args.model, custom_objects={
            'squeeze': squeeze}
                                           )
    else:
        if args.model_type :
            model = models.get_model(args.model_type, output_dim=len(LABELS), features_dim=feature_shape )
        else :
            model = models.conv_net_lstm_attention(output_dim=len(LABELS), features_dim=feature_shape)

        model.summary()

        if args.checkpoint_path:
            model.load_weights(args.checkpoint_path)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LR_INIT,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    optimizer_adam = tf.keras.optimizers.Adagrad(learning_rate=LR_INIT)

    # Define our metrics

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
    validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

    model.compile(optimizer=optimizer_adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    model.fit(dataset_train,
              epochs=EPOCHS,
              validation_data=dataset_validation,
              callbacks=callbacks)

    print("Finished training the model... \n")
    print("Saving the model....")
    model.save('logs/final_model.h5')

    print("Running test metrics")

    dataset_test = tf.data.Dataset.from_tensor_slices(format_dataset(testset)).batch(1)

    test_loss, test_acc = model.evaluate(dataset_test)

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))
