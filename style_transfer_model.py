import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


class StyleTransferModel:
    """
    A class to apply and run style transfer for a Content Image from a given Style Image
    Attributes
    ----------
        model : Hub Module
        content : Numpy Array
        style : Numpy Array
        stylized : Tensorflow Eager Tensor

    Methods
    -------
        load(image_path) :
            Loads an image as a numpy array and normalizes it from the given image path
        stylize_image(content_path,style_path) :
            Applies Neural Style Transfer to Content Image from Style Image and Displays the Stylized Image
    """

    def __init__(self):
        """
        Constructs the Fast Arbitrary Image Style Transfer Model from Tensorflow Hub

        Parameters
        ----------
            None
        """
        # Fast arbitrary image style transfer model from Tensorflow Hub
        self.model = hub.load(
            'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        self.content = None
        self.style = None
        self.stylized = None

    @staticmethod
    def load(image, target_size=(512, 512)):
        """
        Loads an image as a numpy array and normalizes it from the given image path

        Parameters
        ----------
            image :
                File path of the image or bytes file image

        Returns
        -------
            img : Numpy Array
            :param image:
            :param target_size:
        """
        if isinstance(image, bytes):
            img = tf.io.decode_image(image)

        else:
            img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.array([img / 255.0])
        return img

    def stylize_image(self, content_image, style_image, size_image=(512, 512)) -> Image:
        """
        Applies Neural Style Transfer to Content Image from Style Image and Displays the Stylized Image

        Parameters
        ----------
            content_path : str
                path of the Content Image
            style_path : str
                path of the Style Image

        Returns
        -------
            Image
            :param content_image:
            :param style_image:
            :param size_image:
        """
        try:
            self.content = self.load(content_image, size_image)
            self.style = self.load(style_image, size_image)
            self.stylized = self.model(tf.image.convert_image_dtype(self.content, tf.float32),
                                       tf.image.convert_image_dtype(self.style, tf.float32))[0]

            # self.show_stylize_image()
            return self.tensor_to_image(self.stylized[0])
        except Exception as e:
            print("Error Occurred :", e)

    def show_stylize_image(self):
        if self.stylized[0] is not None:
            plt.figure(figsize=(10, 10))

            plt.subplot(1, 3, 1)
            plt.imshow(self.content[0])
            plt.title('Content Image')

            plt.subplot(1, 3, 2)
            plt.imshow(self.style[0])
            plt.title('Style Image')

            plt.subplot(1, 3, 3)
            plt.imshow(self.stylized[0])
            plt.title('Stylized Image')
            plt.show()

    def save_stylized_image(self, image_path_name):
        stylized_image = self.tensor_to_image(self.stylized)

        stylized_image.save(image_path_name)

    @staticmethod
    def tensor_to_image(tensor) -> Image:
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)


if __name__ == '__main__':
    model = StyleTransferModel()
    model.stylize_image("/home/home/Документы/image_ml/model_stylize/thumbnails/style0.jpg",
                        "/home/home/Документы/image_ml/model_stylize/thumbnails/style1.jpg",
                        (512, 512))
    model.save_stylized_image("image.jpeg")
