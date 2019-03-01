import mrcnn.model as modellib
from config import PageConfig
import click
from dataset.dataset import PageDataset
from dataset.dataset import ICDAR_convert



def train(epochs, save_dir, data_dir, collapse=False):
    collapse = bool(collapse)
    # just in case
    if data_dir[-1] != '/':
        data_dir += '/'
    data_train = PageDataset('train', data_dir, collapse)
    data_train.load_page(classes=list(ICDAR_convert.keys()))
    data_train.prepare()
    data_val = PageDataset('val', data_dir, collapse)
    data_val.load_page(classes=list(ICDAR_convert.keys()))
    data_val.prepare()
    config = PageConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=save_dir)
    try:
        model_path = model.find_last()
        print("reloading wieghts from {}".format(model_path))
    except Exception:
        model_path = model.get_imagenet_weights()
    model.load_weights(model_path, by_name=True, exclude=['mrcnn_bbox_fc', 'mrcnn_class_logits', 'mrcnn_mask'])

    model.train(data_train, data_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')

@click.command()
@click.argument("epochs",type=click.INT)
@click.argument("save_dir")
@click.argument("data_dir")
@click.option("--collapse", default=False)
def main(*inputs,**kwargs):
    train(*inputs,**kwargs)


if __name__ == "__main__":
    main()
