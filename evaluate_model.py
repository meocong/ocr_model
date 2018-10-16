import click
import os
import cv2
import glob
import numpy as np

from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

from model.img2seq_ctc import Img2SeqCtcModel
from model.img2seq import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.image import greyscale

def imgshow(img):
    plt.imshow(img)
    plt.show()

def flatten_list(lst):
    while(type(lst) is list):
        lst = lst[0]

    return lst

def read_img(path, _img_prepro=greyscale, padding=False):
    #mostly copied from model/utils/data_generator
    assert os.path.exists(path)

    # load and normalize image
    img = cv2.imread(path)
    if padding: img = add_padding(img)

    img = _img_prepro(img)

    return [img]

def read_imgs(path, _img_prepro=greyscale, padding=False):
    assert os.path.exists(path)
    imgs = []

    if path[-1] == '/': path = path[:-1]
    for _path in glob.glob("%s/*" % path):
        img = read_img(_path, _img_prepro=greyscale, padding=padding)
        imgs += img

    return imgs

def init_model(model_saved_path, vocab_path, is_ctc = True):
    # currently support CTC version and Attention

    # load parameters
    config_data = Config(model_saved_path + "data.json")
    config_vocab = Config(model_saved_path + "vocab.json")
    config_model = Config(model_saved_path + "model.json")

    config_vocab.path_vocab = vocab_path

    vocab = Vocab(config_vocab)

    if is_ctc:
        model = Img2SeqCtcModel(config_model, model_saved_path, vocab)
    else:
        model = Img2SeqModel(config_model, model_saved_path, vocab)

    model.build_pred()
    model.restore_session(model_saved_path + "model.weights/")

    return model

def predict_with_image(model_saved_path, vocab_path, imgs, is_ctc=False):
    model = init_model(model_saved_path, vocab_path, is_ctc)

    hyps = []
    for img in imgs:
        hyp = model.predict_batch(images=[img])[0]
        hyps += [flatten_list(hyp)]

    return hyps

def predict_with_path(model_saved_path, vocab_path, img_path, is_dir=True, is_ctc=False):

    # load image or folder of images
    if is_dir: imgs = read_imgs(path=img_path)
    else: imgs = read_img(path=img_path)

    return predict_with_image(model_saved_path,vocab_path,imgs, is_ctc)

def add_padding(image, padding = 12, padding_left = 12):
    temp = np.ones((image.shape[0] + padding * 2, image.shape[1] + padding + padding_left, image.shape[2]), np.uint8) * 255
    temp[padding:padding + image.shape[0], padding_left:padding_left + image.shape[1], :] = image[:, :, :]
    return temp

def insert_image_worksheet(worksheet,
                           img_path, index, col_index, scale=3, positioning=2):
    cell_width = 64.0
    cell_height = 20.0

    img = np.array(Image.open(img_path))
    height = img.shape[0]
    worksheet.insert_image(scale*index-scale+1, col_index, img_path, {
        'x_offset': 2, 'y_offset': 2,
        # 'x_scale': 20.0/height, 'y_scale': 20.0/height,
        'x_scale': scale*20.0/height, 'y_scale': scale*20.0/height,
        'positioning': positioning
    })

if __name__ == "__main__":
    #predicts = predict_with_path(model_saved_path='./results_from_server/results1/full/', vocab_path="./results_from_server/results1/full/vocab.txt", img_path='./test_imgs/en1_5.jpg', is_dir=False)

    predicts = predict_with_path(model_saved_path='./results1/full/',
                                 vocab_path="./results1/full/vocab.txt",
                                 img_path='./test_imgs/en1_0.jpg', is_dir=False, is_ctc=False)

    for predict in predicts:
        print (predict)