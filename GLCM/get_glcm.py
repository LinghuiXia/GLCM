# coding: utf-8
# The code is written by Linghui

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage import data
from math import floor, ceil
from skimage.feature import greycomatrix, greycoprops

def main():
    pass

def image_patch(img2, slide_window, h, w):

    image = img2
    window_size = slide_window
    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)

    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            patch[:, :, i, j] = img2[i : i + slide_window, j : j + slide_window]
 
    return patch

def calcu_glcm(img, vmin=0, vmax=255, nbit=64, slide_window=5, step=[2], angle=[0]):
        
    mi, ma = vmin, vmax
    h, w = img.shape

    # Compressed gray range：vmin: 0-->0, vmax: 256-1 -->nbit-1
    bins = np.linspace(mi, ma+1, nbit+1)
    img1 = np.digitize(img, bins) - 1

    # (512, 512) --> (slide_window, slide_window, 512, 512)
    img2 = cv2.copyMakeBorder(img1, floor(slide_window/2), floor(slide_window/2)
                              , floor(slide_window/2), floor(slide_window/2), cv2.BORDER_REPLICATE) # 图像扩充

    patch = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)
    patch = image_patch(img2, slide_window, h, w)

    # Calculate GLCM (5, 5, 512, 512) --> (64, 64, 512, 512)
    # greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
    glcm = np.zeros((nbit, nbit, len(step), len(angle), h, w), dtype=np.uint8)
    for i in range(patch.shape[2]):
        for j in range(patch.shape[3]):
            glcm[:, :, :, :, i, j]= greycomatrix(patch[:, :, i, j], step, angle, levels=nbit)

    return glcm

def calcu_glcm_mean(glcm, nbit=64):
    '''
    calc glcm mean
    '''
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    return mean

def calcu_glcm_variance(glcm, nbit=64):
    '''
    calc glcm variance
    '''
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean)**2

    return variance

def calcu_glcm_homogeneity(glcm, nbit=64):
    '''
    calc glcm Homogeneity
    '''
    Homogeneity = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            Homogeneity += glcm[i,j] / (1.+(i-j)**2)

    return Homogeneity

def calcu_glcm_contrast(glcm, nbit=64):
    '''
    calc glcm contrast
    '''
    contrast = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            contrast += glcm[i, j] * (i-j)**2

    return contrast

def calcu_glcm_dissimilarity(glcm, nbit=64):
    '''
    calc glcm dissimilarity
    '''
    dissimilarity = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            dissimilarity += glcm[i, j] * np.abs(i-j)

    return dissimilarity

def calcu_glcm_entropy(glcm, nbit=64):
    '''
    calc glcm entropy 
    '''
    eps = 0.00001
    entropy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            entropy -= glcm[i, j] * np.log10(glcm[i, j] + eps)

    return entropy

def calcu_glcm_energy(glcm, nbit=64):
    '''
    calc glcm energy or second moment
    '''
    energy = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            energy += glcm[i, j]**2

    return energy

def calcu_glcm_correlation(glcm, nbit=64):
    '''
    calc glcm correlation (Unverified result)
    '''
    
    mean = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i, j] * i / (nbit)**2

    variance = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            variance += glcm[i, j] * (i - mean)**2
    
    correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            correlation += ((i - mean) * (j - mean) * (glcm[i, j]**2))/variance

    return correlation

def calcu_glcm_Auto_correlation(glcm, nbit=64):
    '''
    calc glcm auto correlation
    '''
    Auto_correlation = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            Auto_correlation += glcm[i, j] * i * j

    return Auto_correlation


if __name__ == '__main__':
    main()
    
