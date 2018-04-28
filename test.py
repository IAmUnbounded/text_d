import numpy as np
from PIL import Image
import click
import glob
from model import generator_model
from utils import load_images, deprocess_image
import cv2
RESHAPE = (256,256)
def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def load_image(images):
    images_n = []
    for im in images:
        #print(im)
        img = Image.open(im)
        images_n.append(preprocess_image(img))
    return np.array(images_n)

def test(batch_size):
    #data = load_images('./images/test', batch_size)
    y_train = sorted(glob.glob('/home/turing/td/data/*.png'))
    x_train = sorted(glob.glob('/home/turing/td/blur/*.png'))
    y_test, x_test = load_image(y_train[:5]), load_image(x_train[:5])
    g = generator_model()
    g.load_weights('weights1/428/generator_13_261.h5')
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    generated = np.array([deprocess_image(img) for img in generated_images])
    x_test = deprocess_image(x_test)
    y_test = deprocess_image(y_test)

    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated[i, :, :, :]
        #print img.shape
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #y = cv2.cvtColor(y,cv2.COLOR_BGR2GRAY)
        #x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        output = np.concatenate((y, x, img), axis=1)
        im = Image.fromarray(output.astype(np.uint8))
        im.save('results{}.png'.format(i))


@click.command()
@click.option('--batch_size', default=1, help='Number of images to process')
def test_command(batch_size):
    return test(batch_size)


if __name__ == "__main__":
    test_command()
