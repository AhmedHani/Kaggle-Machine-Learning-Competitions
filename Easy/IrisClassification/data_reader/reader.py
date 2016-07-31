import csv


class CsvReader(object):
    def __init__(self, file_path):
        self.__file_path = file_path

    def get_iris_data(self):
        f = open(self.__file_path, 'rb')
        reader = csv.reader(f)

        iris_features = []
        iris_labels = []
        index = 0

        for row in reader:
            if index == 0:
                index += 1
                continue

            iris_features.append([row[1], row[2], row[3], row[4]])
            iris_labels.append(row[5])

        f.close()

        return iris_features, iris_labels