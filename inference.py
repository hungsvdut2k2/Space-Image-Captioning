import tensorflow as tf
import numpy as np
from data.build_dataset import BuildDataset
from models.image_captioning_model import ImageCaptioningModel
from argparse import ArgumentParser


def inference(config: dict):
    build_dataset = BuildDataset(
        config["caption_file_path"],
        config["max_sequence_length"],
        config["vocab_size"],
        config["image_size"],
        config["batch_size"],
    )
    model = ImageCaptioningModel(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["n_heads"],
        dff=config["d_ff"],
        vocab_size=config["vocab_size"],
        dropout_rate=config["dropout_rate"],
        model_name=config["model_name"],
    )
    model.load_weights(config["checkpoint_path"])

    image = build_dataset.process_input_image(config["image_path"])
    display_image = image.numpy().clip(0, 255).astype(np.uint8)
    image = np.expand_dims(display_image, axis=0)

    vectorizer = build_dataset.create_vectorizer()
    vocab = vectorizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))

    output_str = "<start>"
    for index in range(config["vocab_size"]):
        tokenized_caption = vectorizer([output_str])[:, :-1]
        pred = model.predict((image, tokenized_caption), verbose=0)
        sampled_token_idx = np.argmax(pred[0, index, :])
        sampled_token = index_lookup[sampled_token_idx]
        if sampled_token == "<end>":
            break
        output_str += f" {sampled_token}"
    generated_caption = output_str.replace("<start>", "")
    generated_caption = generated_caption.replace("<end>", "")
    generated_caption = generated_caption.strip()

    return display_image, generated_caption


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--caption-file-path", type=str)
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    args = parser.parse_args()

    config = {
        "caption_file_path": args.caption_file_path,
        "model_name": args.model_name,
        "num_layers": args.num_layers,
        "dropout_rate": 0.2,
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 8,
        "image_size": args.image_size,
        "max_sequence_length": 30,
        "vocab_size": 5000,
        "train_size": 0.7,
        "val_size": 0.2,
        "image_path": args.image_path,
        "check_point_path": args.check_point_path,
    }
    display_image, generated_caption = inference(config)
    print(generated_caption)
