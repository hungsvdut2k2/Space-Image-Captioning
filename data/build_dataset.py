import tensorflow as tf
import numpy as np
from .data_normalize import text_normalize


class BuildDataset:
    def __init__(
        self,
        caption_file_path: str,
        max_sequence_length: int,
        vocab_size: int,
        image_size: int,
        batch_size: int,
    ):
        self.caption_file_path = caption_file_path
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.dataset_dict = {}
        self.corpus = []
        self.image_size = (image_size, image_size)
        self.batch_size = batch_size

    def init_dataset(self):
        with open(self.caption_file_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_path, caption = line.split("\t")

                tokens = caption.split(" ")

                if len(tokens) < 5 or len(tokens) > (self.max_sequence_length - 2):
                    continue
                else:
                    normalized_caption = text_normalize(caption)
                    self.corpus.append(normalized_caption)
                    self.dataset_dict[img_path] = normalized_caption

    def augment(self):
        augmented_layer = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomContrast(0.3),
            ]
        )
        return augmented_layer

    def create_vectorizer(self):
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_sequence_length,
            standardize=None,
        )
        vectorizer.adapt(self.corpus)
        return vectorizer

    def preprocess_img(self, image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, self.image_size)
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    def process_input_image(
        self,
        image_path: str,
        caption: str,
        vectorizer: tf.keras.layers.TextVectorization,
        augmented=False,
    ):
        image = self.preprocess_img(image_path)
        if augmented:
            image = self.augment(image)
        caption = vectorizer(caption)
        decoder_input = caption[:-1]
        decoder_output = caption[1:]

        return (image, decoder_input), decoder_output

    def build_dataset(self, train_size: float, val_size: float):
        self.init_dataset()
        vectorizer = self.create_vectorizer()

        x = np.array(list(self.dataset_dict.keys()))
        y = np.array(list(self.dataset_dict.values()))

        train_index = int(len(x) * train_size)
        val_index = train_index + int(len(x) * val_size)
        indexes = np.arange(len(x))
        indexes = np.random.permutation(indexes)

        x = x[indexes]
        y = y[indexes]
        x_train, y_train = x[:train_index], y[:train_index]
        x_val, y_val = x[train_index:val_index], y[train_index:val_index]
        x_test, y_test = x[val_index:], y[val_index:]

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))

        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        train_ds = (
            train_ds.cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .map(
                lambda x, y: self.process_input_image(
                    x, y, vectorizer, augmented=False
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(self.batch_size)
        )

        val_ds = (
            val_ds.cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .map(
                lambda x, y: self.process_input_image(
                    x, y, vectorizer, augmented=False
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(self.batch_size)
        )

        test_ds = (
            test_ds.cache()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
            .map(
                lambda x, y: self.process_input_image(
                    x, y, vectorizer, augmented=False
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(self.batch_size)
        )
        return train_ds, val_ds, test_ds
