MAX_TRIES = 300


class MatrixOperation:
    @staticmethod
    def multiply(first_multiplier, second_multiplier):
        if type(second_multiplier[0]) == int:
            return list(list(
                sum([first_multiplier[i][k] * second_multiplier[k] for k in range(len(second_multiplier))]) for i in
                range(len(first_multiplier))))
        else:
            return list(list(
                sum([first_multiplier[i][k] * second_multiplier[k][j] for k in range(len(second_multiplier))]) for j in
                range(len(second_multiplier[0]))) for i in range(len(first_multiplier)))

    @staticmethod
    def transposition(matrix):
        return list(list(matrix[j][i] for j in range(len(matrix))) for i in range(len(matrix[0])))

    @staticmethod
    def multiply_number(number, matrix):
        return list(list(matrix[i][j] * number for j in range(len(matrix[0]))) for i in range(len(matrix)))


class TemplatesManager:
    def __init__(self):
        self.file_name = ['M','T','A']
        self.pattern = self.convert_from_file_to_pattern(self.file_name)
        self.example = self.convert_from_file_to_example()

    def convert_from_file_to_pattern(self,file_name):
        text = []
        pattern = []
        for name in file_name:
            with open(f"templates/{name}", "r") as f:
                text.append(f.read())
        for letter in text:
            letter = letter.replace('.', '-1 ').replace('*', '1 ').replace('\n', '')[:-1]
            letter = [int(el) for el in letter.split(' ')]
            pattern.append(letter)
        return pattern


    def convert_from_file_to_example(self):
        with open("example","r") as f:
            example = f.read()
        example = example.replace('.', '-1 ').replace('*', '1 ').replace('\n', '')[:-1]
        example = [int(el) for el in example.split(' ')]
        return example



class Network:
    def __init__(self, patterns):
        self.patterns = patterns
        self.weight_matrix_size = len(self.patterns)
        self.weight = [[0 for _ in range(self.weight_matrix_size)] for _ in range(self.weight_matrix_size)]

    @staticmethod
    def sign(element):
        return 1 if element >= 0 else -1

    @staticmethod
    def print(example):
        result = ''
        for i in range(len(example)):
            if i % 9 == 0:
                result += '\n'
            result += str(example[i])
        print(result.replace('-1', '.').replace('1', '*'))

    def learning(self, example):
        cnt = 0
        while example not in self.patterns and cnt<MAX_TRIES:
            cnt += 1
            self.weight = MatrixOperation.multiply(MatrixOperation.transposition(self.patterns), self.patterns)
            self.weight = MatrixOperation.multiply_number(1 / self.weight_matrix_size, self.weight)
            for i in range(self.weight_matrix_size):
                self.weight[i][i] = 0
            example = MatrixOperation.multiply(self.weight, example)
            example = [Network.sign(element) for element in example]
        print('Error, check your input in "example"')  if cnt>=300 else Network.print(example)

if __name__ == '__main__':
    manager = TemplatesManager()
    patterns = manager.pattern
    example = manager.example
    print("It's your output:")
    Network.print(example)
    print("\nSome magic 🪄🪄🪄")
    mine = Network(patterns)
    mine.learning(example)

