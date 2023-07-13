# Space-Image-Captioning

## How to use

## Clone this repository

```$bash
$ git clone https://github.com/hungsvdut2k2/Space-Image-Captioning
```

## Go into the repository

```$bash
$ cd Space-Image-Captioning
```

## Install packages

```bash
$ pip install -r requirements.txt
```

## Set up your dataset

```bash
$ python crawler/crawler.py --{num_pages}
```

## Train your model by running this command line

```bash
$ python train.py --caption-file-path{caption_file_path} --batch-size{batch_size} --epochs{num_epochs} --model-name{model_name}
```

## Inference using this command line

```bash
$ python inference.py --caption-file-path{caption_file_path} --image-path{image_path} --check-point-path{check_point_path} --model-name{model_name}
```
