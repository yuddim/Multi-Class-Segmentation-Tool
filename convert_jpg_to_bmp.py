import os
import cv2
import numpy as np
import main

obj_palete = [(0, 200, 0), (200, 0, 0), (0, 0, 200)]


def binary_to_color_with_pallete(binary_im, pallete_color):

    #im_reshaped = binary_im[:, :, 0]
    #im_reshaped = im_reshaped.reshape((im_reshaped.shape[0], im_reshaped.shape[1], 1))
    im_reshaped = binary_im
    im_reshaped = im_reshaped.reshape((im_reshaped.shape[0], im_reshaped.shape[1], 1))/255
    im_out = np.append(im_reshaped * (255 - pallete_color[0]), im_reshaped * (255 - pallete_color[1]), axis=2)
    im_out = np.append(im_out, im_reshaped * (255 - pallete_color[2]), axis=2)
    im_out = 255 - im_out
    return im_out
#Загрузка изображения на форму
def load_image_file(file_name):

    img0 = cv2.imread(file_name, 1)

    lower_range = np.array([0, 0, 0], dtype=np.uint8)
    upper_range = np.array([200, 200, 200], dtype=np.uint8)
    # Маска изображения, выделяющая пятно:
    mask = cv2.inRange(img0, lower_range, upper_range).astype('uint8')
    #mask = (img0 != (255,255,255)).astype('uint8')
    new_image = binary_to_color_with_pallete(mask, obj_palete[0])

    return new_image

file_dir_jpg = "D:/Datasets/Students_monitoring/images/markup"
file_dir_bmp = os.path.dirname(file_dir_jpg)
if(file_dir_bmp !=''):
    file_mk_dir = file_dir_bmp + '/markup_bmp'
    os.makedirs(file_mk_dir, exist_ok=True)

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    filenames = []

    for filename in sorted(os.listdir(file_dir_jpg)):
        is_valid = False
        for extension in white_list_formats:
            if filename.lower().endswith('.' + extension):
                is_valid = True
                break
        if is_valid:
            filenames.append(filename)


    for f_name in filenames:
        image_file = file_dir_jpg + '/'+ f_name
        new_image = load_image_file(image_file)

        res_fname, res_extension = os.path.splitext(image_file)
        res_fname = os.path.basename(res_fname)
        target_fname_markup = file_mk_dir + '/' + res_fname + '.bmp'
        cv2.imwrite(target_fname_markup, new_image)
        print (target_fname_markup)