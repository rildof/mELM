import os
import cv2
import copy
import numpy as np
import pandas as pd
import preprocess_images as PI
from skimage.feature import graycomatrix, graycoprops

def main_generate(image_dir):
    # Lista para armazenar os resultados
    results = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            features = PI.calcular_descritores_haralick(image_path)
            features.update(PI.calcular_porcentagens(image_path))  
            results.append(features)

    
    df = pd.DataFrame(results)
    print(f'Salvando dataset_termografia_mama.csv')
    df.to_csv('dataset_termografia_mama.csv', index=False)
    print('Processo finalizado!')

def main_preprocess(input_directory, output_directory, limits_list):
    image_path = PI.listar_imagens(input_directory)
    print(image_path)
    y_limits = limits_list[0]
    x_limits = limits_list[1]
    for path in image_path:        
        img = PI.reduce_image(cv2.imread(path), y_limits, x_limits)
        img = PI.remover_cores(img)
        path_new_directory = output_directory + path.split('\\')[-1].split('.')[0] + '_preprocessada.jpg'
        PI.salvar_imagem(img, path_new_directory)
    


if __name__ == '__main__':

    input_directory =  'diretorio de entrada das imagens aqui' #input_directory = "TCC\ImagensdeTermografiadeMamaSeparadasPorOrientacao\LEMD"
    output_directory = 'diretorio de saida das imagens aqui'#output_directory = "TCC\ImagensdeTermografiadeMamaSeparadasPorOrientacao\LEMD_preprocessada\\"
    main_preprocess(input_directory=input_directory, output_directory=output_directory, limits_list=[(60, 360), (50, 570)])    
    main_generate(output_directory)
   