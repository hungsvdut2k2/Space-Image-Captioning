import tensorflow as tf 
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from data.build_dataset import BuildDataset
from models.image_captioning_model import ImageCaptioningModel
from scheduler import CustomSchedule
from utils import masked_accuracy, masked_loss

def train_model(config: dict):
    build_dataset = BuildDataset(
        config['caption_file_path'], 
        config['max_sequence_length'], 
        config['vocab_size'], 
        config['image_size'], 
        config['batch_size']
    )
    train_ds, val_ds, test_ds = build_dataset.build_dataset(config['train_size'], config['val_size'])
    model = ImageCaptioningModel(
        num_layers=config['num_layers'], 
        d_model=config['d_model'],
        num_heads=config['n_heads'],
        dff=config['d_ff'],
        vocab_size=config['vocab_size'],
        dropout_rate=config['dropout_rate'],
        model_name=config['model_name']
    )
    learning_rate = CustomSchedule(config['d_model'])
    optimizer = tf.keras.optimizers.Adam(
            learning_rate, 
            beta_1=0.9, 
            beta_2=0.98,
            epsilon=1e-9
    )
    model.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy]
    )
    history = model.fit(
        train_ds,
        epochs=config['epochs'],
        validation_data=val_ds
    )
    test_evaluation = model.evaluate(test_ds)
    model.save("./weight", save_format="tf")
    train_loss, train_acc = history.history['loss'], history.history['masked_accuracy']
    val_loss, val_acc = history.history['val_loss'], history.history['val_masked_accuracy']
    return (train_loss, train_acc), (val_loss, val_acc)

def visualize(train_history, val_history):
    train_loss, train_accuracy = train_history
    val_loss, val_accuracy = val_history
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.plot(train_loss, color='green')

    plt.subplot(2, 2, 2)
    plt.xlabel('Epochs') 
    plt.ylabel('Accuracy')
    plt.title('Training accuracy') 
    plt.plot(train_accuracy, color='orange')

    plt.subplot(2, 2, 3)
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.title('Validation loss')
    plt.plot(val_loss, color='green')

    plt.subplot(2, 2, 4)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy')
    plt.plot(val_accuracy, color='orange')

    plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--caption-file-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--model-name", type=str)
    parser.add_argument(
        "--is-augmented", type=lambda x: (str(x).lower() == "true"), default=False
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    config = {
        "caption_file_path":args.caption_file_path,
        "batch_size":args.batch_size,
        "epochs":args.epochs,
        "model_name": args.model_name,
        "is_augmented": args.is_augmented,
        "num_layers":args.num_layers,
        "dropout_rate":0.2,
        "d_model":128,
        "d_ff":512,
        "n_heads":8,
        "image_size": args.image_size,
        "max_sequence_length":30,
        "vocab_size":5000,
        "train_size":0.7,
        "val_size":0.2
    }
    train_history, val_history = train_model(config)
    visualize(train_history, val_history)