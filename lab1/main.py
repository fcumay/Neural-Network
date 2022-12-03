from PIL import Image
from random import uniform
import json
import numpy as np
import os.path


class MatrixOperation:
    @staticmethod
    def multiply(first_multiplier, second_multiplier):
        return tuple(tuple(
            sum([first_multiplier[i][k] * second_multiplier[k][j] for k in range(len(second_multiplier))]) for j in
            range(len(second_multiplier[0]))) for i in range(len(first_multiplier)))

    @staticmethod
    def multiply_number(number, matrix):
        return tuple(tuple(matrix[i][j] * number for j in range(len(matrix[0]))) for i in range(len(matrix)))

    @staticmethod
    def subtract(first_multiplier, second_multiplier):
        return tuple(
            tuple(first_multiplier[i][j] - second_multiplier[i][j] for j in range(len(second_multiplier[0]))) for i in
            range(len(first_multiplier)))

    @staticmethod
    def transposition(matrix):
        return tuple(tuple(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix[0])))


class ImageEditor:
    def __init__(self, file_name, user_width, user_height):
        self.img = None
        self.file_name = f"images/{file_name}.bmp"
        self.user_width, self.user_height = user_width, user_height
        self.pixels, self.width, self.height = self.convert_image()
        self.rectangles = self.divide_into_rectangles()
        self.pixels_color = None

    def convert_image(self):
        img = Image.open(self.file_name)
        img = img.convert('RGB')
        self.img = img
        (width, height) = img.size
        pixels = list(img.getdata())
        pixels = tuple(tuple(((x * 2) / 255) - 1 for x in y) for y in pixels)
        return pixels, width, height

    def isDivisionValid(self, user_width, user_height):
        return (self.width * self.height) % (user_width * user_height) == 0

    def divide_into_rectangles(self):
        square = self.user_height * self.user_width
        if self.isDivisionValid(self.user_width, self.user_height):
            rectangles = tuple(self.pixels[i * square: square * (i + 1):] for i in range(0, (
                    len(self.pixels) // square)))
            return rectangles
        else:
            print('Invalid values. Try again. (Ex. 4,4)')
            self.divide_into_rectangles()

    def convert_rectangles_to_vector(self):
        vector_image = []
        vector_rectangle = []
        for i_rectangle in range(0, len(self.rectangles)):
            vector_rectangle.clear()
            for pixel in self.rectangles[i_rectangle]:
                vector_rectangle.extend(pixel)
            vector_image.append(tuple(vector_rectangle))
        return tuple(vector_image)

    def convert_to_matrix(self, vector):
        color = []
        matrix_rectangle = []
        matrix = []
        for rectangle in vector:
            matrix_rectangle.clear()
            for i in range(0, len(rectangle), 3):
                color.clear()
                for j in range(0, 3):
                    color.append(rectangle[i + j])
                matrix_rectangle.append(tuple(color))
            matrix.append(tuple(matrix_rectangle))
        return tuple(matrix)

    def convert_to_pixel(self, matrix):
        pixels_color = []
        pixels = []
        for rectangle in matrix:
            pixels.extend(rectangle)
        pixels_color = tuple(tuple(int((1 + x) / 2 * 255) for x in y) for y in pixels)
        self.pixels_color = pixels_color

    def show_pic(self):
        out_image = Image.new(self.img.mode, self.img.size)
        out_image.putdata(self.pixels_color)
        out_image.show()


class Studying:
    def __init__(self, image=None, error=None, neurons=3):
        self.neurons = neurons
        self.image = image
        self.weight1 = self.weights()[0]
        self.weight2 = self.weights()[1]
        self.layer1 = self.expected = image.convert_rectangles_to_vector()
        self.layer2 = self.layer3 = None
        self.error = error

    def weights(self):
        weight1 = tuple(tuple(uniform(-1, 1) for _ in range(self.neurons)) for __ in
                        range(self.image.user_width * self.image.user_height * 3))
        weight2 = tuple(
            tuple(uniform(-1, 1) for _ in range(self.image.user_width * self.image.user_height * 3)) for __ in
            range(self.neurons))
        return weight1, weight2

    def errors(self, delta):
        return sum([j ** 2 for i in delta for j in i])

    def alpha(self, matrix):
        return 1 / sum([j ** 2 for i in matrix for j in i])

    def normolize(self):
        transpose_weight1 = MatrixOperation.transposition(self.weight1)
        transpose_weight2 = MatrixOperation.transposition(self.weight2)
        buf1 = tuple(sum(transpose_weight1[i][j] ** 2 for j in range(len(transpose_weight1[i]))) for i in
                     range(len(transpose_weight1)))
        buf2 = tuple(sum(transpose_weight2[i][j] ** 2 for j in range(len(transpose_weight2[i]))) for i in
                     range(len(transpose_weight2)))
        self.weight1 = tuple(tuple(self.weight1[i][j] ** 2 / buf1[j] for j in range(len(self.weight1[i]))) for i in
                             range(len(self.weight1)))
        self.weight2 = tuple(tuple(self.weight2[i][j] ** 2 / buf2[j] for j in range(len(self.weight2[i]))) for i in
                             range(len(self.weight2)))

    def start_studying(self):
        error = self.error + 1
        while error > self.error:
            print('Study\n')
            # new_layer =  (layer_i * Wi)
            self.layer2 = MatrixOperation.multiply(self.layer1, self.weight1)
            self.layer3 = MatrixOperation.multiply(self.layer2, self.weight2)
            # delta = (actual_value - expected_value)
            delta = MatrixOperation.subtract(self.layer3, self.layer1)
            error = self.errors(delta)
            # W2(t+1) = W2 - a` * layer2^T * delta
            self.weight2 = MatrixOperation.subtract(self.weight2, MatrixOperation.multiply(
                MatrixOperation.multiply_number(self.alpha(self.layer2), MatrixOperation.transposition(self.layer2)),
                delta))
            # W1(t+1) = W1 - a * layer1^T * delta  * W2 ^ T
            self.weight1 = MatrixOperation.subtract(self.weight1, MatrixOperation.multiply(
                MatrixOperation.multiply_number(self.alpha(self.layer2), MatrixOperation.transposition(self.layer1)),
                MatrixOperation.multiply(delta, MatrixOperation.transposition(self.weight2))))
            # self.normolize()
            print('Yep', error)

        return self.layer3

    def save_weight(self):
        to_json = {'user_width': self.image.user_width, 'user_height': self.image.user_height, 'weight1': self.weight1,
                   'weight2': self.weight2}
        with open(f'storage{self.image.file_name[6:-4:]}.json', 'w+') as f:
            f.write(json.dumps(to_json))

    def save_weight_to_json(self):
        if os.path.exists(f'storage/{self.image.user_width}.json'):
            if self.error > self.upload_weight(self.image.user_width)[4]:
                return None
        to_json = {'error': self.error, 'user_width': self.image.user_width, 'user_height': self.image.user_height,
                   'weight1': self.weight1,
                   'weight2': self.weight2}
        with open(f'storage/{self.image.user_width}.json', 'w+') as f:
            f.write(json.dumps(to_json))

    @staticmethod
    def upload_weight(name):
        with open(f'storage/{name}.json') as f:
            data = f.read()
        data = json.loads(data)
        return data['user_width'], data['user_height'], data['weight1'], data['weight2'], data['error']


if __name__ == '__main__':
    choice = int(input(
        'Choose mode:\n1)Quick start --> Using\n2)Upload weights from storage --> Using\n3)Quick start --> Learning\n4)New weights --> Learning \n'))
    if choice == 1:
        storage_name = '2'
        image_name = '2'
        (user_width, user_height, weight1, weight2, error) = Studying.upload_weight(storage_name)
        error = 1900
    elif choice == 2:
        storage_name = input('Input name from storage: (Ex. 4)\n')
        image_name = input('Input image name: (Ex. ъуъ)\n')
        (user_width, user_height, weight1, weight2, error) = Studying.upload_weight(storage_name)
        error = int(input('Input value of the admissible errors: (Ex. 9000)'))
    elif choice == 3:
        image_name = 'ъуъ'
        user_width = 4
        user_height = 4
        error = 100000
    elif choice == 4:
        image_name = input('Input image name: (Ex. 3)\n')
        user_width = int(input('Input wight: (Ex. 4)\n'))
        user_height = int(input('Input height: (Ex. 4)\n'))
        error = int(input('Input value of the admissible errors: (Ex. 60000)'))

    image1 = ImageEditor(image_name, int(user_width), int(user_height))
    student = Studying(image1, error)
    if choice == 1 or choice == 2:
        student.weight1 = weight1
        student.weight2 = weight2

    image1.convert_to_pixel(image1.convert_to_matrix(student.start_studying()))
    student.save_weight_to_json()
    image1.show_pic()
