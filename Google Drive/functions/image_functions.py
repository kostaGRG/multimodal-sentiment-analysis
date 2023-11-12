from torchvision import transforms
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor,
                                    ToPILImage,
                                    Lambda)
from tqdm.notebook import tqdm
from sentiment_analysis_functions import get_batches
import torch
from transformers import ViTImageProcessor, BeitImageProcessor
import numpy as np
from PIL import Image

# Process the image, applying the transform (chosen image augmentations) 
# and passing the image through the image processor
def process_image(image,transform,imageProcessor):
    transformed_image = transform(image)
    tensor_image = imageProcessor(transformed_image, return_tensors='pt')['pixel_values'][0]
    return tensor_image

# Add gaussian noise to the input image, with standard deviation=stddev
def add_gaussian_noise(img,stddev):
    np_img = np.array(img)
    noise = np.random.normal(scale=stddev, size=np_img.shape)
    noisy_img = np_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# Transform all the images of the dataset, using the image processor
# and applying image augmentations if the boolean IMAGE_ROTATIONS is TRUE.
# In the output, the transformed images are returned with their corresponding labels.
def transform_images(images,labels,imageProcessor,image_mean,image_std, size, IMAGE_ROTATIONS=True,batch_size=16,stddev=0.1):
    processed_images = []
    
    if IMAGE_ROTATIONS:
        transform = Compose(
                [
                    ToPILImage(),
                    RandomResizedCrop(size),
                    RandomHorizontalFlip(),
                    Lambda(lambda img: add_gaussian_noise(img,stddev)),
                    ToTensor(),
                    Normalize(mean=image_mean, std=image_std),
                    ToPILImage()
                ]
            )
    else:
        transform = Compose(
                [
                    ToPILImage()
                ]
            )

    for batch in tqdm(get_batches(images, batch_size)):
        batch_tensors = [process_image(image,transform,imageProcessor) for image in batch]
        processed_images.extend(batch_tensors)

    images = torch.stack(processed_images)
    labels = torch.tensor(labels)

    return images,labels

# Calculate mean, standard deviation and size that imageProcessor applies to the images
def get_processor_stats(imageProcessor):
    image_mean = imageProcessor.image_mean
    image_std = imageProcessor.image_std
    size = imageProcessor.size["height"]
    return image_mean, image_std, size  

# Choose the appropriate image processor based on the selected model
def choose_processor(MODEL):
    if MODEL == 'microsoft/beit-base-patch16-224-pt22k-ft22k':
        imageProcessor = BeitImageProcessor.from_pretrained(MODEL)
    else:
        imageProcessor = ViTImageProcessor.from_pretrained(MODEL)
    image_mean, image_std, size = get_processor_stats(imageProcessor)
    return imageProcessor, image_mean, image_std, size