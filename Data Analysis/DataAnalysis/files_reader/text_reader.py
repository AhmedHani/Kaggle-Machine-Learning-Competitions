__author__ = 'Ahmed Hani Ibrahim'


class TextReader(object):
    def __init__(self, file):
        if not file.endswith('.txt'):
            raise ValueError('This is not a .txt file. Check file extension please!')

        self.__file = file
        self.__data = open(file, 'r')
        self.__lines = []

    def get_data(self):
        for line in self.__data:
            self.__lines.append(line)

        return self.__lines

    def get_data_size(self):
        return len(self.__lines)
