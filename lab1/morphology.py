"""Бинаризация и две реализации одной итерации дилатации 3×3."""

import cv2
import numpy as np


def validate_binary(image: np.ndarray) -> None:
    """Принимаются только непустые двумерные uint8-маски со значениями 0/255."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Изображение должно быть массивом NumPy.")
    if image.ndim != 2 or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Ожидается непустое двумерное изображение типа uint8.")
    if not np.all((image == 0) | (image == 255)):
        raise ValueError("Бинарное изображение должно содержать только 0 и 255.")


def binarize(gray: np.ndarray, threshold: int = 127, invert: bool = False) -> np.ndarray:
    """Порог OpenCV: I > threshold становится белым; invert меняет полярность."""
    if gray.ndim != 2 or gray.size == 0 or gray.dtype != np.uint8:
        raise ValueError("Ожидается непустое полутоновое изображение типа uint8.")
    if not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError("Порог должен быть целым числом от 0 до 255.")
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(gray, threshold, 255, mode)[1]


def dilate_opencv(image: np.ndarray) -> np.ndarray:
    """Квадрат 3×3, центральный якорь, одна итерация, нулевая граница."""
    validate_binary(image)
    return cv2.dilate(
        image,
        np.ones((3, 3), dtype=np.uint8),
        anchor=(1, 1),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def dilate_native(image: np.ndarray) -> np.ndarray:
    """Поиск максимума в каждом окне 3×3 обычными циклами Python.

    NumPy используется для проверки и преобразования формата данных,
    но не для вычисления дилатации. Исходное изображение не изменяется.
    """
    validate_binary(image)
    height, width = image.shape
    source = [[0] * (width + 2)]
    source.extend([0] + row + [0] for row in image.tolist())
    source.append([0] * (width + 2))
    result = [[0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            maximum = 0
            for dy in range(3):
                row = source[y + dy]
                for dx in range(3):
                    value = row[x + dx]
                    if value > maximum:
                        maximum = value
            result[y][x] = maximum

    return np.array(result, dtype=np.uint8)
