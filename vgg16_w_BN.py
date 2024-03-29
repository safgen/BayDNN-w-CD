import os
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.utils import get_source_inputs, to_categorical
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, \
    BatchNormalization, Activation, Dropout
from keras import backend as K, optimizers
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from keras import datasets
from concrete_dropout import ConcreteDropout


def conv_block(units, dropout=0.2, activation='relu', block=1, layer=1):
    def layer_wrapper(inp):
        x = Conv2D(units, (3, 3), padding='same', name='block{}_conv{}'.format(block, layer))(inp)
        x = BatchNormalization(name='block{}_bn{}'.format(block, layer))(x)
        x = Activation(activation, name='block{}_act{}'.format(block, layer))(x)
        x = Dropout(dropout, name='block{}_dropout{}'.format(block, layer))(x)
        return x

    return layer_wrapper


def dense_block(units, dropout=0.2, activation='relu', name='fc1', concrete_dropout=False):
    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        if not concrete_dropout:
            x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        else:
            x = ConcreteDropout(Dense(units))(x)
        return x

    return layer_wrapper


def VGG16_BN(input_tensor=None, input_shape=None, classes=10, conv_dropout=0.4, dropout=0.5, activation='relu'):
    """Instantiates the VGG16 architecture with Batch Normalization
    # Arguments
        input_tensor: Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: shape tuple
        classes: optional number of classes to classify images
    # Returns
        A Keras model instance.
    """
    img_input = Input(shape=input_shape) if input_tensor is None else (
        Input(tensor=input_tensor, shape=input_shape) if not K.is_keras_tensor(input_tensor) else input_tensor
    )

    # Block 1
    x = conv_block(64, dropout=0.3, activation=activation, block=1, layer=1)(img_input)
    x = conv_block(64, dropout=0, activation=activation, block=1, layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = conv_block(128, dropout=conv_dropout, activation=activation, block=2, layer=1)(x)
    x = conv_block(128, dropout=0, activation=activation, block=2, layer=2)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=1)(x)
    x = conv_block(256, dropout=conv_dropout, activation=activation, block=3, layer=2)(x)
    x = conv_block(256, dropout=0, activation=activation, block=3, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=1)(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=4, layer=2)(x)
    x = conv_block(512, dropout=0, activation=activation, block=4, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=1)(x)
    x = conv_block(512, dropout=conv_dropout, activation=activation, block=5, layer=2)(x)
    x = conv_block(512, dropout=0, activation=activation, block=5, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Flatten
    x = GlobalAveragePooling2D()(x)

    # FC Layers
    x = dense_block(512, dropout=dropout, activation=activation, name='fc1')(x)
    x = dense_block(512, dropout=dropout, activation=activation, name='fc2')(x)

    # Classification block
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    inputs = get_source_inputs(input_tensor) if input_tensor is not None else img_input

    # Create model.
    return Model(inputs, x, name='vgg16_bn')


def get_vgg16bn_with_concrete_dropout(model):
    """
    Changes Dropout in VGG16 architecture with Batch Normalization to Concrete Dropout
        # Arguments
            A keras vgg16bn model instance
        # Returns
            A Keras model instance.
    """
    # modify layers for concrete dropout
    x = model.get_layer(name='block5_pool').output
    # Flatten
    x = GlobalAveragePooling2D()(x)

    # FC Layers
    x = dense_block(512, concrete_dropout=True, activation='relu', name='fc1_cd')(x)
    x = dense_block(512, concrete_dropout=True, activation='relu', name='fc2_cd')(x)

    # Classification block
    x = Dense(10, activation='softmax', name='predictions_cd')(x)
    return Model(inputs=model.input, outputs=x, name='vgg16_bn_cd')


ADAM_OPTIM = True  # Toggle to switch between adam and SGD optimizers
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# print("******************")
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Convert class vectors to binary class matrices using one hot encoding
y_train_ohe = to_categorical(y_train, num_classes=10)
y_test_ohe = to_categorical(y_test, num_classes=10)

# Data normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# print("******************")
# print(X_train.shape)
# print(y_train_ohe.shape)
# print(X_test.shape)
# print(y_test_ohe.shape)


# PRETRAINING WITH DROPOUT

# initiate optimizer
if ADAM_OPTIM:
    opt = optimizers.adam()
else:
    opt = optimizers.SGD(lr=0.001, momentum=0.9)

# For a multi-class classification problem
model = VGG16_BN(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

es = EarlyStopping(patience=5, monitor='val_accuracy', mode='max')
mc = ModelCheckpoint('./vgg.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# initialize the number of epochs and batch size
EPOCHS = 100
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True, fill_mode="nearest")

if os.path.exists('./vgg.h5'):
    model.load_weights('./vgg.h5')
# train the model
PT_history = model.fit_generator(aug.flow(X_train, y_train_ohe, batch_size=BS), validation_data=(X_test, y_test_ohe),
                                 steps_per_epoch=len(X_train) // BS, epochs=EPOCHS, callbacks=[es, mc])

# We load the best weights saved by the ModelCheckpoint
model.load_weights('./vgg.h5')

# ------------------------------------------------------------------------------
# PREDICT AND EVALUATE THE PRETRAINED NETWORK
# ------------------------------------------------------------------------------
preds = model.predict(X_test)
score_test = accuracy_score( y_test, np.argmax(preds, axis=1))
print ('Test Score : ', score_test)


train_loss, train_score = model.evaluate(X_train, y_train_ohe)
test_loss, test_score = model.evaluate(X_test, y_test_ohe)
# print("Train Loss:", train_loss)
# print("Test Loss:", test_loss)

# File for printing results
res = sample = open('results.txt', 'a')
print("Training accuracy after pretraining:", train_score, file=res)
print("Test accuracy after pretraining:", test_score,  file=res)

# TRAINING THE CNN ON THE TRAIN/VALIDATION DATA FOR FINE TUNING

# get the model with concrete dropout layers
model_FT = get_vgg16bn_with_concrete_dropout(model=model)
model_FT.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

mc = ModelCheckpoint('./vgg_FT.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# train the model
FT_history = model_FT.fit_generator(aug.flow(X_train, y_train_ohe, batch_size=BS), validation_data=(X_test, y_test_ohe),
                                    steps_per_epoch=len(X_train) // BS, epochs=EPOCHS, callbacks=[es, mc])

# We load the best weights saved by the ModelCheckpoint
model_FT.load_weights('./vgg_FT.h5')

# ------------------------------------------------------------------------------
# PREDICT AND EVALUATE THE NETWORK ON THE TEST DATA AFTER FINE TUNING
# ------------------------------------------------------------------------------
preds = model_FT.predict(X_test)
score_test = accuracy_score( y_test, np.argmax(preds, axis=1) )
print ('Test Score : ', score_test)


train_loss, train_score = model_FT.evaluate(X_train, y_train_ohe)
test_loss, test_score = model_FT.evaluate(X_test, y_test_ohe)
# print("Train Loss:", train_loss)
# print("Test Loss:", test_loss)
print("Train accuracy after fine-tuning with concrete dropout:", train_score, file=res)
print("Test accuracy after fine-tuning with concrete dropout:", test_score, file=res)
res.close()
