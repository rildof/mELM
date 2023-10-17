import os
import cv2
import copy
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def listar_imagens(diretorio):
    directory_paths = []
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            caminho_completo = os.path.join(root, file)
            directory_paths.append(caminho_completo)
    return directory_paths

def filtrar_bilateral(image, d=12, sigma_color=75, sigma_space=75):
    # Aplicar o filtro bilateral
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return filtered_image

def filtrar_mediano(image, kernel_size=9):
    # Aplicar o filtro mediano
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image

def filtrar_gaussiano(image, kernel_size=11, sigma=5):
    # Aplicar o filtro gaussiano
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    return filtered_image

def remover_cores(image):
    image = filtrar_bilateral(image=image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir os limites para a remoção de cores
    blue = [np.array([80, 50, 50]), np.array([150, 255, 255])]
    purple = [np.array([130, 0, 0]), np.array([170, 255, 255])]
    green = [np.array([35, 50, 50]), np.array([90, 255, 255])]

    # Criação das máscaras para cada cor
    mask_blue = cv2.inRange(hsv_image, blue[0], blue[1])
    mask_green = cv2.inRange(hsv_image, green[0], green[1])
    mask_purple = cv2.inRange(hsv_image, purple[0], purple[1])
    # Aplicação da máscara à imagem original
    
    filtred_img = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask_blue))
    filtred_img = cv2.bitwise_and(filtred_img, image, mask=cv2.bitwise_not(mask_purple))
    filtred_img = cv2.bitwise_and(filtred_img, image, mask=cv2.bitwise_not(mask_green))
    return filtred_img
 
def reduce_image(img: np.ndarray, altura_limits: tuple, largura_limits: tuple) -> np.ndarray:
    return img[altura_limits[0]:altura_limits[1],largura_limits[0]:largura_limits[1]]

def salvar_imagem(imagem, caminho_completo):
    cv2.imwrite(caminho_completo, imagem)
    print("Imagem salva com sucesso!")

def calcular_porcentagens(image_path):
    # Definir os limites das cores amarelo, vermelho e branco no formato BGR
    limites_cores = [
        {'cor': 'amarelo', 'limite': [(20, 50, 50), (40, 255, 255)]},
        {'cor': 'vermelho', 'limite': [(0, 50, 50), (10, 255, 255)]},
        {'cor': 'branco', 'limite': [(0, 0, 200), (255, 30, 255)]}
    ]

    imagem = cv2.imread(image_path)
    # Converter a imagem de BGR para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Calcular a quantidade total de pixels
    total_pixels = imagem.size / 3

    # Calcular as porcentagens de pixels para cada cor
    cores_porcentagens = {}
    for limite_cor in limites_cores:
        mascara = cv2.inRange(hsv, limite_cor['limite'][0], limite_cor['limite'][1])
        pixels_cor = np.sum(mascara) / 255
        porcentagem_cor = (pixels_cor / total_pixels) * 100
        cores_porcentagens[limite_cor['cor']] = f"{round(porcentagem_cor, 2)}%"

    return cores_porcentagens


def calcular_descritores_haralick(image_path):
    # Carregar a imagem em escala de cinza
    img = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_BayerGR2GRAY)

    # Calcular a matriz de co-ocorrência de níveis de cinza
    distances = [1, 2, 3, 4]  # Distâncias entre pixels para a co-ocorrência
    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Ângulos de co-ocorrência

    # Calcular a matriz de co-ocorrência
    glcm = graycomatrix(img, distances=distances, angles=theta, levels=256, symmetric=True, normed=True)

    # Calcula os descritores de Haralick
    descritores = {
        'image_name': image_path.split("\\")[-1], 
        'contraste': graycoprops(glcm, prop='contrast').mean(),
        'energia': graycoprops(glcm, prop='energy').mean(),
        'homogeneidade': graycoprops(glcm, prop='homogeneity').mean(),
        'correlacao': graycoprops(glcm, prop='correlation').mean(),
        'ASM': graycoprops(glcm, prop='ASM').mean(),
        'variancia': graycoprops(glcm, prop='ASM').var(),
        'diferenca_entropia': graycoprops(glcm, prop='dissimilarity').mean(),
        'diferenca_variancias': graycoprops(glcm, prop='ASM').mean() - np.var(img),
    }

    return descritores