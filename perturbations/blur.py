from PIL import ImageFilter


def gaussian_blur(image, radius):
    if radius < 0:
        raise ValueError("radius must be >= 0")

    return image.filter(
        ImageFilter.GaussianBlur(radius=radius)
    )