import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from wandb.keras import WandbCallback
from tqdm.notebook import tqdm



def train_model(model, 
                training_data,
                validation_data,
                name='my_model.h5',
                stopping_contidion='val_accuracy',
                steps_per_epoch=150,
                verbose=False,
                epochs=16):
    ' procedure to train model using fixed setting and callbacks '

    # callback - save the model at each epoch end
    checkpoint = ModelCheckpoint(name,
                                 monitor=stopping_contidion,   # metric to monitor
                                 verbose=True,
                                 save_best_only=True,)

    # callback - stop training when the model reaches a platau
    early = EarlyStopping(monitor=stopping_contidion,          # metric to monitor
                          patience=6, 
                          verbose=True)

    log_dir = 'logs'
    wandb_callback = WandbCallback(
                       predictions=10)

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'],
                  optimizer='Adam')

    history = model.fit(training_data,
                        validation_data=validation_data,
                        callbacks=[checkpoint, early, wandb_callback],
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        verbose=verbose,
                        workers=1)
    
    return tf.keras.models.load_model(name)


from skimage import io, morphology, transform, feature, segmentation, measure, color
from matplotlib import cm

def get_labels(pred, shuffle=False):
    
    # find local maxima
    im = morphology.erosion(pred, morphology.disk(3))
    peak_idx = feature.peak_local_max(im, min_distance=7, threshold_rel=.5)
    markers = np.zeros_like(pred, dtype=bool)
    markers[tuple(peak_idx.T)] = True

    # segment
    labels = segmentation.watershed(-pred, measure.label(markers), mask=pred>.1)
        
    return labels



def save_results(res, store_dir='results', img_dir='BF-C2DL-HSC/test'):

    # get input images
    img_names = os.listdir(img_dir)
    img_names.sort()

    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)

    for i in tqdm(range(len(res))):
        
        # read image
        img_path = os.path.join(img_dir, img_names[i] )
        img = io.imread(img_path)
        img = transform.resize(img, (1000, 1000))
        
        # get prediction
        pred = res[i, :, :, 1]

        # get labels
        labels = get_labels(pred)
        store_path = os.path.join(store_dir, f'res{i+1700:04d}.jpg')

        # visialize
        cmap = cm.get_cmap('tab20b').copy()
        cmap.set_bad(color='black')

        color_mask = color.label2rgb(labels,
                               colors=cmap.colors,
                               image=img,
                               bg_label=0,
                               alpha=.4)

        # save result
        io.imsave(store_path, (color_mask*255).astype(np.uint8), check_contrast=False)
        


def show_predictions(model,
                     dataset,
                     idx=0):
    
    # get index of the batch and index of the image in the batch
    batch_size = next(iter(dataset))[0].shape[0]
    img_idx, batch_idx = idx % batch_size, idx // batch_size
    
    # take input batch
    for im, gt in dataset.take(batch_idx + 1):
        pass
    
    # predict batch
    res = model.predict(im)

    # get input and result
    labels = ['input image', 'ground truth', 'prediction']   
    images = [inverse_preprocess_input(im[img_idx, :, :]),
              gt[img_idx, :, :],
              np.argmax(res[img_idx, :, :], axis=-1)]

    # plot result
    plot_in_grid(images, labels, n_cols=3, title=f' Image index: {str(idx)}')
    
    
def show_augmentation(dataset, idx, n_samples=8):
    
    batch_size = next(iter(dataset))[0].shape[0]
    img_idx, batch_idx = idx % batch_size, idx // batch_size
    
    images, gts = [], []

    for i in range(n_samples):
        for im, gt in dataset.take(batch_idx):
            pass

        # de-preprocess image vgg16
        img = im[img_idx, :, :].numpy()
        img = inverse_preprocess_input(img)

        images.append( img)
        images.append(gt[img_idx, :, :])

    plot_in_grid(images, [])


def set_dynamic_gpu_memory_allocation():
    gpus = tf.config.list_physical_devices('GPU')
    print(f'Detected gpus: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
def inverse_preprocess_input(img):
    # rescale
    img = img + np.array([[[ 103.939, 116.779, 123.68 ]]])
           
    # cut off overflowing values
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    
    # BGR to RGB
    r, g, b = np.split(img, 3, axis=-1)
    img = np.concatenate([b, g, r], axis=2)
    
    return img


def ishow(img,
          cmap='viridis',
          title='',
          fig_size=(8,6),
          colorbar=True,
          interpolation='none'):
    ' Function `ishow` displays an image in a new window. '
    
    extent = (0, img.shape[1], img.shape[0], 0)
    fig, ax = plt.subplots(figsize=fig_size)
    pcm = ax.imshow(img,
              extent=extent,
              cmap=cmap,
              interpolation=interpolation)
    
    ax.set_frame_on(False)
    plt.title(title)
    plt.tight_layout()
    if colorbar:
        
        fig.colorbar(pcm, orientation='vertical')
    plt.show()
        
    
# old
        
        
def plot_training_history(train_log):
    ' plot training history '
    names = train_log.history.keys()

    for name in names:
        plt.plot(train_log.history[name])

    plt.title("training history")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(names)
    plt.show()
        
        
def plot_in_grid(images,
                 names,
                 n_cols=4,
                 title='',
                 cmap='viridis',
                 fixed_range=False,
                 colorbar=False,
                 pad_images=False):
    '''
    Plots grid of images
    
    : param images : list of numpy arrays
    : param names : list of strings, names of figures
    : param n_cols : n cols in a grid
    : param title : string of the whle figure name
    : param fixed_range : False or a tuple (min_value, max_value)
    '''
        
    if len(images) != len(names):
        names = [''] * len(images)
    
    n_samples = len(images)
    n_rows = int(np.ceil(n_samples / 4)) 
    
    fig, axes = plt.subplots(figsize=(15,n_rows*4),
                         nrows=n_rows,
                         ncols=n_cols)
    fig.suptitle(title, fontsize=12)

    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(images, names)):
        if fixed_range:
            vmin, vmax = fixed_range
        else:
            vmin, vmax = np.min(img), np.max(img)
        if pad_images:
            img = np.pad(img, ((1, 1), (1, 1)))
        extent = (0, img.shape[1], img.shape[0], 0)
        pcm = ax[i].imshow(img,
                           cmap=cmap,
                           interpolation="none",
                           vmin=vmin,
                           vmax=vmax,
                           extent=extent)
        ax[i].set_title(title, fontsize=10)
        
        if colorbar:
            fig.colorbar(pcm, ax=ax[i], shrink=0.75, location='bottom')

    plt.tight_layout()
    plt.show()
    


def plot_experiment(x,
                    y,
                    legend,
                    x_label='kernel radius (px)',
                    y_label='Average execution time (s)',
                    title='Execution time / Kernel radius'):
    ''' visualize execution time 
    : param x : list of parameters
    : param y : list measurements
    : param legend : names of measurements
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    for sample in y:
        plt.plot(x, sample)
        plt.scatter(x, sample)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend)
    plt.grid()
    plt.show()
    
    
def plot_batch(batch, labels, n_rows=1):
    '''
    plots several samples fro mthe batch
    : param : image data, batch of N samples
    : label : np array of size N
    : n_rows : number of ploted rows of lenght 4
    '''
    fig, axes = plt.subplots(figsize=(15,4*n_rows),
                         nrows=n_rows,
                         ncols=4)
    ax = axes.ravel()
    
    for i in range(n_rows*4):
        img = batch[i, :, :]
        lbl = labels[i]
        ax[i].imshow(img, 'gray', interpolation="none")
        ax[i].set_title(f'label: {lbl}\nintensity range:  {np.min(img)} - {np.max(img)}', fontsize=12)

    plt.tight_layout()
    plt.show()
    

def compare_images(img, ref):
    ' compares intensities of two images at pixel level '
    diff = np.abs(img-ref)
    if np.max(diff) < np.e ** -10:
        print('COMPARE: Images are the same.')
    else:
        print(f'COMPARE: Images differ. ')
        plt.imshow(diff, interpolation='none')
        plt.colorbar()
        plt.show()

