########################################################################################################################
########################################################################################################################
# DATA: 19/08/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# E-MAIL: vinicius.carneiro@ufla.br
# GITHUB: vqcarneiro
### grupo: Everton Cardoso, Julio Miguel, Lorena Dumbá e Reberth Silva
########################################################################################################################
################################## TRABALHO DE VISÃO COMPUTACIONAL - ESCURECIMENTO DE GRÃOS ###########################
#######################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt



epoca1 = ('CARIOCAMG T18 R1 P98.jpg', 'CARIOCAMG T18 R2 P213.jpg','CARIOCAMG T18 R3 P293.jpg', 'MADREPEROLA T42 R1 P09.jpg', 'MADREPEROLA T42 R2 P147.jpg',
          'MADREPEROLA T42 R3 P194.jpg','UAI T39 R1 P06.jpg', 'UAI T39 R2 P193.jpg', 'UAI T39 R1 P06.jpg', 'MAJESTOSO T59 R1 P56.jpg', 'MAJESTOSO T59 R2 P175.jpg','MAJESTOSO T59 R3 P247.jpg',
          'PEROLA T90 R1 P79.jpg', 'PEROLA T90 R2 P164.jpg', 'PEROLA T90 R3 P272.jpg')
epoca2 = ('Escurecimento CARIOCAMG T18 R1 P98.png', 'Escurecimento CARIOCAMG T18 R2 P213.png','Escurecimento CARIOCAMG T18 R3 P293.png',
          'Escurecimento MADREPEROLA T42 R1 P09.png', 'Escurecimento MADREPEROLA T42 R2 P147.png','Escurecimento MADREPEROLA T42 R3 P294.png',
          'Escurecimento UAI T39 R1 P06.png', 'Escurecimento UAI T39 R2 P193.png','Escurecimento UAI T39 R3 P362.png','Escurecimento MAJESTOSO T59 R1 P56.png',
          'Escurecimento MAJESTOSO T59 R2 P175.png','Escurecimento MAJESTOSO T59 R3 P247.png', 'Escurecimento PEROLA T90 R1 P79.png', 'Escurecimento PEROLA T90 R2 P164.png','Escurecimento PEROLA T90 R3 P272.png')


vars = np.zeros((15,3))
it=0

for i, c in zip(epoca1, epoca2):
    img_bgr = cv2.imread(i, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    (L, img_otsu) = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_segmentada = cv2.bitwise_and(img_rgb, img_rgb, mask=img_otsu)
    hist_sementes_r = cv2.calcHist([img_segmentada], [0], img_otsu, [256], [0, 256])


    img_bgr2 = cv2.imread(c, 1)
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
    img_lab2 = cv2.cvtColor(img_rgb2, cv2.COLOR_RGB2LAB)
    l2, a2, b2 = cv2.split(img_lab2)
    (L2, img_otsu2) = cv2.threshold(b2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_segmentada2 = cv2.bitwise_and(img_rgb2, img_rgb2, mask=img_otsu2)
    hist_sementes_r2 = cv2.calcHist([img_segmentada2], [0], img_otsu2, [256], [0, 256])

    plt.figure('Análise')
    plt.subplot(1, 4, 1)
    plt.imshow(img_segmentada)
    plt.xticks([])
    plt.yticks([])
    plt.title(i)

    plt.subplot(1, 4, 2)
    plt.plot(hist_sementes_r)
    ymax = max(hist_sementes_r)
    xmax, _ = np.where(hist_sementes_r == ymax)
    plt.legend([xmax], loc=9)


    plt.subplot(1, 4, 3)
    plt.imshow(img_segmentada2)
    plt.title(c)

    plt.subplot(1, 4, 4)
    plt.plot(hist_sementes_r2)
    ymax2 = max(hist_sementes_r2)
    xmax2, _ = np.where(hist_sementes_r2 == ymax2)
    plt.legend([xmax2], loc=9)
    vars[it,0]=xmax
    vars[it,1]=xmax2
    vars[it,2] =xmax-xmax2

    it=it+1

    #plt.show()


gen=np.arange(1,6,1)
gen=np.repeat(gen,3).reshape(15,1)
rep=np.arange(1,4,1)
rep=np.concatenate((rep, rep, rep, rep, rep), axis=0)
rep=rep.reshape(15,1)

tab_geral=np.concatenate((gen, rep, vars), axis=1)

print(tab_geral)
np.savetxt("tabelagerada_trabalho visao.txt", tab_geral, fmt="%.2f" , delimiter=" ", header= "Genótipos, rep, epoca1, epoca2, diferença")

genotipos=np.arange(1,6,1).reshape(5,1)
mingen=np.zeros((5,1))
maxgen= np.zeros((5,1))
meangen=np.zeros((5,1))
vargen=np.zeros((5,1))


it=0
for el in np.arange(0,15,3):
    maxgen[it,0] = np.max(tab_geral[el:el + 3, 4], axis=0)
    mingen[it, 0] = np.min(tab_geral[el:el + 3, 4], axis=0)
    meangen[it,0] = np.mean(tab_geral[el:el + 3, 4], axis=0)
    vargen[it,0] = np.var(tab_geral[el:el + 3, 4], axis=0)
    it += 1

print("-----------------------------------")
print("Tabela: genotipos, max, min, media da diferença, variancia ")

matriz_resumo = np.concatenate((genotipos, maxgen, mingen, meangen, vargen), axis=1)

print(matriz_resumo)
np.savetxt("tabelagerada_trabalho_resumo.txt", matriz_resumo, fmt="%.1f" , delimiter=" ", header= "Genótipos, rep, epoca1, epoca2, diferença")
