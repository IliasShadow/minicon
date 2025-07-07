import numpy as np
from PIL import Image


def ordered_dithering(image, matrix):
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    m_height, m_width = matrix.shape
    output = np.zeros_like(img_array)
    for y in range(height):
        for x in range(width):
            threshold = matrix[y % m_height, x % m_width]
            if img_array[y, x] > threshold:
                output[y, x] = 255
            else:
                output[y, x] = 0
    return Image.fromarray(output.astype(np.uint8), mode='L')


def floyd_steinberg_dithering(image):
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    output = np.copy(img_array).astype(np.int16)
    for y in range(height):
        for x in range(width):
            old_pixel = output[y, x]
            new_pixel = 255 if old_pixel > 127 else 0
            output[y, x] = new_pixel
            error = old_pixel - new_pixel
            if x + 1 < width:
                output[y, x + 1] = np.clip(output[y, x + 1] + error * 7 // 16, 0, 255)
            if y + 1 < height:
                if x - 1 >= 0:
                    output[y + 1, x - 1] = np.clip(output[y + 1, x - 1] + error * 3 // 16, 0, 255)
                output[y + 1, x] = np.clip(output[y + 1, x] + error * 5 // 16, 0, 255)
                if x + 1 < width:
                    output[y + 1, x + 1] = np.clip(output[y + 1, x + 1] + error * 1 // 16, 0, 255)
    return Image.fromarray(output.astype(np.uint8), mode='L')


def create_lut(brightness=1.0, saturation=1.0):
    lut = []
    for i in range(256):
        value = i * brightness
        value = max(0, min(255, value))
        lut.append(int(value))
    return lut


def apply_lut_rgb(image, lut):
    r, g, b = image.split()
    r = r.point(lut)
    g = g.point(lut)
    b = b.point(lut)
    return Image.merge("RGB", (r, g, b))


def apply_lut_hsv(image, lut):
    hsv_image = image.convert('HSV')
    h, s, v = hsv_image.split()
    v = v.point(lut)
    hsv_image = Image.merge("HSV", (h, s, v))
    return hsv_image.convert('RGB')


if __name__ == "__main__":
    input_image_path = "input_image.jpg"
    output_image_path = "output.png"
    image = Image.open(input_image_path)
    print("Выполняется полутонирование...")
    dither_matrix = np.array([[0, 128], [192, 64]])
    ordered_dithered = ordered_dithering(image, dither_matrix)
    ordered_dithered.save("ordered_dithered.png")
    print("Результат сохранен в 'ordered_dithered.png'.")
    floyd_steinberg_dithered = floyd_steinberg_dithering(image)
    floyd_steinberg_dithered.save("floyd_steinberg_dithered.png")
    print("Результат сохранен в 'floyd_steinberg_dithered.png'.")
    print("Выполняется коррекция цвета...")
    brightness_lut = create_lut(brightness=1.5)
    saturation_lut = create_lut(saturation=1.5)
    rgb_corrected = apply_lut_rgb(image, brightness_lut)
    rgb_corrected.save("rgb_corrected.png")
    print("Коррекция цвета в RGB завершена. Результат сохранен в 'rgb_corrected.png'.")
    hsv_corrected = apply_lut_hsv(image, saturation_lut)
    hsv_corrected.save("hsv_corrected.png")
    print("Коррекция цвета в HSV завершена. Результат сохранен в 'hsv_corrected.png'.")