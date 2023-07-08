import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


def get_cnn_model(model_params: dict, model_name: str):
    model_dict = {
        "vgg": tf.keras.applications.vgg16.VGG16(**model_params),
        "resnet": tf.keras.applications.resnet50.ResNet50(**model_params),
        "inception_resnet": tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
            **model_params
        ),
    }
    base_model = model_dict[model_name]

    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = tf.keras.layers.Reshape((-1, base_model_out.shape[-1]))(
        base_model_out
    )
    cnn_model = tf.keras.models.Model(base_model.input, base_model_out)

    return cnn_model


class ImageCaptioningModel(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        vocab_size,
        dropout_rate=0.1,
        model_name
    ):
        super().__init__()
        model_params = {
            "input_shape": (*(128, 128), 3),
            "include_top": False,
            "weights": "imagenet",
        }
        self.cnn_model = get_cnn_model(model_params, model_name)
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
        )

        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        img, x = inputs

        context = self.cnn_model(img)

        context = self.encoder(context)

        x = self.decoder(x, context)

        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
