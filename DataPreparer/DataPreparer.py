import os.path
import uuid
import random
import uuid
import xlsxwriter

from PIL import Image, ImageDraw
import pandas


class DataPreparer:
    def __init__(self, element_count, data_path, out_path, leave_if_prepared=True):
        self.element_count = element_count

        if leave_if_prepared and os.path.isfile(out_path + '/data.csv'):
            return

        bit_map = Image.open(fp=data_path)

        x_data = []
        z_data = []
        width_data = []
        height_data = []
        file_name = []
        width = 20 / 128
        height = 20 / 128
        for buff in range(element_count):
            x_n = random.randint(0, 108)
            z_n = random.randint(0, 108)
            x_data.append(x_n/128)
            z_data.append(z_n/128)
            height_data.append(height)
            width_data.append(width)
            image = Image.new(mode='RGB', size=(128, 128), color=(0, 0, 0))
            image.paste(im=bit_map, box=(x_n, z_n))
            to_append = str(uuid.uuid4()) + '.png'
            numa = out_path + '/' + to_append
            image.save(fp=numa, format='PNG')
            file_name.append(to_append)

        data_frame = pandas.DataFrame({'x_data': x_data,
                                       'z_data': z_data,
                                       'width': width_data,
                                       'height': height_data,
                                       'file_name': file_name})

        data_frame.to_csv(out_path + '/data.csv', index=False)

    def getElementCount(self):
        return self.element_count
