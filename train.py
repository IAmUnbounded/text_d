import os
import datetime
import click
import numpy as np
from utils import load_images
from losses import wasserstein_loss, perceptual_loss
from model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs
import glob
from keras.optimizers import Adam
from PIL import Image
from keras.models import load_model

BASE_DIR = 'weights1/'
RESHAPE = (256,256)
def preprocess_image(cv_img):
    cv_img = cv_img.resize(RESHAPE)
    img = np.array(cv_img)
    img = (img - 127.5) / 127.5
    return img

def load_batch(images):
    images_n = []
    for im in images:
        #print(im)
        img = Image.open(im)
        images_n.append(preprocess_image(img))
    return np.array(images_n)

def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates=5):
    #data = load_images('/home/turing/td/', n_images)
    y_train = sorted(glob.glob('/home/turing/td/data/*.png'))
    x_train = sorted(glob.glob('/home/turing/td/blur/*.png'))
    print('loaded_data')
    g = generator_model()
    g.load_weights('weights/424/generator_19_290.h5')
    d = discriminator_model()
    d.load_weights('weights/424/discriminator_19.h5')
    
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    for epoch in range(epoch_num):
        print('epoch: {}/{}'.format(epoch, epoch_num))
        print('batches: {}'.format(len(x_train) / batch_size))

        permutated_indexes = np.random.permutation(len(x_train))

        d_losses = []
        d_on_g_losses = []
        for index in range(int(len(x_train)/ batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            x_t = []
            y_t = []
            for i in batch_indexes:
                x_t.append(x_train[i])
                y_t.append(y_train[i])
            image_blur_batch = load_batch(x_t)
            image_full_batch = load_batch(y_t)
            

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)
            print('batch {} d_loss : {}'.format(index+1, np.mean(d_losses)))

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)
            print('batch {} d_on_g_loss : {}'.format(index+1, d_on_g_loss))

            d.trainable = True

        with open('log.txt', 'a') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=1000000, help='Number of images to load for training')
@click.option('--batch_size', default=16, help='Size of batch')
@click.option('--epoch_num', default=20, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
