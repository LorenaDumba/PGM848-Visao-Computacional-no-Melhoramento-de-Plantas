########################################################################################################################
########################################################################################################################
# DATA: 12/08/2020
# DISCIPLINA: VISÃO COMPUTACIONAL NO MELHORAMENTO DE PLANTAS
# PROFESSOR: VINÍCIUS QUINTÃO CARNEIRO
# E-MAIL: vinicius.carneiro@ufla.br
# GITHUB: vqcarneiro
########################################################################################################################
########################################################################################################################
#     REO 03 - LISTA DE EXERCÍCIOS 3
########################################################################################################################
# EXERCÍCIO 01:
################################################################################################
# Importar pacotes:
import cv2 # Importa o pacote opencv
from matplotlib import pyplot as plt # Importa o pacote matplotlib
import numpy as np
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
################################################################################################
# Leitura da imagem:
nome_arquivo = 'CARIOCAMG T18 R1 P98.jpg' # determina a variável para carregar a imagem
img_bgr = cv2.imread(nome_arquivo,1) # carrega a imagem no sistema bgr e depois converte para RGB.
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB) # faz a conversão para a RGB.

################################################################################################
print('-'*95)
print('1-Selecione uma imagem a ser utilizada no trabalho prático e realize os seguintes processos utilizando\n'
      'as bibliotecas OPENCV e Scikit-Image do Python:')

print('a) Aplique o filtro de média com cinco diferentes tamanhos de kernel e compare os resultados\n'
      'com a imagem original; ')

# CONFECÇÃO DOS FILTROS DE MÉDIA: nesse caso, são feitos filtros de média alterando o valor do kernel
# isso corresponde a uma alteração no número de pixels vizinhos que são considerados para calcular o kernel
# como consequência isso afetará os ruídos e a nitidez da imagem.

img_fm_1 = cv2.blur(img_rgb,(5,5))

img_fm_2 = cv2.blur(img_rgb,(11,11))

img_fm_3 = cv2.blur(img_rgb,(21,21))

img_fm_4 = cv2.blur(img_rgb,(31,31))

img_fm_5 = cv2.blur(img_rgb,(41,41))

################################################################################################
# APRESENTANDO AS IMAGENS NA MATPLOTLIB:
###############################################################################################

plt.figure('Filtros') # para gerar uma figura
plt.subplot(2,3,1) # Essa figura tem duas linhas e três colunas.
plt.imshow(img_rgb) # imagem original.
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(2,3,2)
plt.imshow(img_fm_1) # originou o kernel 5x5.
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("5x5")

plt.subplot(2,3,3)
plt.imshow(img_fm_2) # kernel 11x11.
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("11x11")

plt.subplot(2,3,4)
plt.imshow(img_fm_3) #Kernel 21x21
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("21x21")

plt.subplot(2,3,5)
plt.imshow(img_fm_4) #Kernel 31x31
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("31x31")

plt.subplot(2,3,6)
plt.imshow(img_fm_5) #Kernel 41x41
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("41x41")

plt.show()
'''
print('-'*95)
print('b) Aplique diferentes tipos de filtros com pelo menos dois tamanhos de kernel e compare os\n'
      'resultados entre si e com a imagem original.')

# APLICANDO DIFERENTES FILTROS:
# I-FILTRO DE MÉDIA:
img_filtro_media1 = cv2.blur(img_rgb,(51,51)) #informa numa tupla o tamanho do kernel 51x51.
img_filtro_media2 = cv2.blur(img_rgb,(31,31)) #informa numa tupla o tamanho do kernel 31x31.

# II-FILTRO GAUSSIANO: faz uma média ponderada os mais próximos tem peso maior.
img_filtro_gaussiano1 = cv2.GaussianBlur(img_rgb,(51,51),0) # Média Ponderada
img_filtro_gaussiano2 = cv2.GaussianBlur(img_rgb,(31,31),0) # Média Ponderada

# III-FILTRO DE MEDIANA: calcula a mediana dos valores que estão na vizinhança. Já existe o pixel. Imagens mais agradáveis.
img_filtro_mediana1 = cv2.medianBlur(img_rgb,51)
img_filtro_mediana2 = cv2.medianBlur(img_rgb,31)

# IV-FILTRO BILATERAL:
img_filtro_bilateral1 = cv2.bilateralFilter(img_rgb,51,51,21)
img_filtro_bilateral2 = cv2.bilateralFilter(img_rgb,31,31,21) # intesidade, distância dos vizinhos.
# ele tem o detalhe para identificar melhor as bordas.
#determina a média ponderada. Elimina os valores de pixels distantes.
#tem que informar o quanto vai eliminar de outilier dessa função.
################################################################################################
# Apresentar imagens no MATPLOTLIB:
plt.figure('Filtro Média')
plt.subplot(3,3,1)
plt.imshow(img_rgb)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("ORIGINAL")

plt.subplot(3,3,2)
plt.imshow(img_filtro_media1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Média 51x51")

plt.subplot(3,3,3)
plt.imshow(img_filtro_media2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Média 31x31")

plt.subplot(3,3,4)
plt.imshow(img_filtro_gaussiano1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Gaussiano 51x51")

plt.subplot(3,3,5)
plt.imshow(img_filtro_gaussiano2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Gaussiano 31x31")

plt.subplot(3,3,6)
plt.imshow(img_filtro_mediana1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Mediana 51x51")

plt.subplot(3,3,7)
plt.imshow(img_filtro_mediana2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Mediana 31x31")

plt.subplot(3,3,8)
plt.imshow(img_filtro_bilateral1)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Bilateral 51x51")

plt.subplot(3,3,9)
plt.imshow(img_filtro_bilateral2)
plt.xticks([]) # Eliminar o eixo X
plt.yticks([]) # Eliminar o eixo Y
plt.title("Bilateral 31x31")

plt.show()
'''
print('-'*95)
print('c) Realize a segmentação da imagem utilizando o processo de limiarização. Utilizando o\n'
      'reconhecimento de contornos, identifique e salve os objetos de interesse. Além disso, acesse\n'
      'as bibliotecas Opencv e Scikit-Image, verifique as variáveis que podem ser mensuradas e\n'
      'extraia as informações pertinentes (crie e salve uma tabela com estes dados). Apresente todas\n'
      'as imagens obtidas ao longo deste processo.')

# CONVERSÃO PARA O CANAL YCrCb:
img_YCrCb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2YCR_CB)
# Partição dos canais
Y,Cr,Cb = cv2.split(img_YCrCb)
# particiona ela nesses canais.
Cb = cv2.medianBlur(Cb,5)
#filtro de mediana
# 9x9 kernel
# Histograma do canal informativo
hist_Cb = cv2.calcHist([Cb],[0], None, [256],[0,256])
# histograma dele filtrado, um canal igual a zero. sem máscara.
########################################################################################################################
# Limiarização - Thresholding
(L, img_limiar_inv) = cv2.threshold(Cb,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#limiarização
########################################################################################################################
# Obtendo imagem segmentada
img_segmentada = cv2.bitwise_and(img_rgb,img_rgb,mask=img_limiar_inv) # usa a mascara da imagem binária
# para ver os grãos coloridos com o fundo removido
########################################################################################################################
# Objetos
# contorno dos grãos de cada objeto da imagem.
mascara = img_limiar_inv.copy()
# gera uma cópia para modificar essa nova que chama mascara
cnts,h = cv2.findContours(mascara, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cv2.RETR_TREE fixo hierarquia
# cv2.CHAIN_APPROX_SIMPLE como obtém os contornos
########################################################################################################################
# Dados Sementes
print('-'*50)
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_properties/py_contour_properties.html

for (i, c) in enumerate(cnts):

	(x, y, w, h) = cv2.boundingRect(c)
	obj = img_limiar_inv[y:y+h,x:x+w]
	obj_rgb = img_segmentada[y:y+h,x:x+w]
	obj_bgr = cv2.cvtColor(obj_rgb,cv2.COLOR_RGB2BGR)
	cv2.imwrite('s'+str(i+1)+'.png',obj_bgr)
	cv2.imwrite('sb'+str(i+1)+'.png',obj)

	regiao = regionprops(obj) #https: // scikit - image.org / docs / dev / api / skimage.measure.html  # skimage.measure.regionprops
	print('Semente: ', str(i+1))
	print('Dimensão da Imagem: ', np.shape(obj))
	print('Medidas Físicas')
	print('Centroide: ', regiao[0].centroid)
	print('Comprimento do eixo menor: ', regiao[0].minor_axis_length)
	print('Comprimento do eixo maior: ', regiao[0].major_axis_length)
	print('Razão: ', regiao[0].major_axis_length / regiao[0].minor_axis_length)
	area = cv2.contourArea(c)
	print('Área: ', area)
	print('Perímetro: ', cv2.arcLength(c,True))

	print('Medidas de Cor')
	min_val_r, max_val_r, min_loc_r, max_loc_r = cv2.minMaxLoc(obj_rgb[:,:,0], mask=obj)
	print('Valor Mínimo no R: ', min_val_r, ' - Posição: ', min_loc_r)
	print('Valor Máximo no R: ', max_val_r, ' - Posição: ', max_loc_r)
	med_val_r = cv2.mean(obj_rgb[:,:,0], mask=obj) # dentro do vermelho 0
	print('Média no Vermelho: ', med_val_r)

	min_val_g, max_val_g, min_loc_g, max_loc_g = cv2.minMaxLoc(obj_rgb[:, :, 1], mask=obj)
	print('Valor Mínimo no G: ', min_val_g, ' - Posição: ', min_loc_g)
	print('Valor Máximo no G: ', max_val_g, ' - Posição: ', max_loc_g)
	med_val_g = cv2.mean(obj_rgb[:,:,1], mask=obj) # dentro do verde 1
	print('Média no Verde: ', med_val_g)

	min_val_b, max_val_b, min_loc_b, max_loc_b = cv2.minMaxLoc(obj_rgb[:, :, 2], mask=obj)
	print('Valor Mínimo no B: ', min_val_b, ' - Posição: ', min_loc_b)
	print('Valor Máximo no B: ', max_val_b, ' - Posição: ', max_loc_b)
	med_val_b = cv2.mean(obj_rgb[:,:,2], mask=obj) # azul 2
	print('Média no Azul: ', med_val_b)
	print('-'*50)

########################################################################################################################
print('Total de sementes: ', len(cnts))
print('-'*50)

seg = img_segmentada.copy()
#cópia da imagem segmentada
cv2.drawContours(seg,cnts,-1,(0,255,0),2)
#desenha um contorno em cada um dos grãos.
# -1 qual grão, indica que faz em todos.
#tupla da cor do contorno R, G, B contorno da cor verde
# 2 espessura do contorno.

plt.figure('Imagem YCrCb')
plt.subplot(2,2,1)
plt.imshow(img_YCrCb)
plt.xticks([])
plt.yticks([])
plt.title('Sementes')

plt.subplot(2,2,2)
plt.plot(hist_Cb)
plt.title('Histograma Cb')

plt.subplot(2,2,3)
plt.imshow(img_limiar_inv)
plt.xticks([])
plt.yticks([])
plt.title('Limiarização')

plt.subplot(2,2,4)
plt.imshow(img_segmentada)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada')

plt.show()

plt.figure('Sementes')
plt.subplot(1,2,1)
plt.imshow(seg)
plt.xticks([])
plt.yticks([])
plt.title('Sementes')

plt.subplot(1,2,2)
plt.imshow(obj_rgb)
plt.xticks([])
plt.yticks([])
plt.title('Semente')


plt.show()

print('-'*95)
print('d) Utilizando máscaras, apresente o histograma somente dos objetos de interesse.')

# Nosso objeto de interesse é o obj_rgb que representa uma semente de feijão. Desse modo,
# fizemos um histograma para cada um dos canais:

hist_segmentada_r = cv2.calcHist([obj_rgb],[0], None, [256],[0,256])
hist_segmentada_g = cv2.calcHist([obj_rgb],[1], None, [256],[0,256])
hist_segmentada_b = cv2.calcHist([obj_rgb],[2], None, [256],[0,256])

plt.figure('Histograma de um Grão')
plt.subplot(1,3,1)
plt.plot(hist_segmentada_r,color = 'red')
plt.title('Histograma Vermelho')
plt.xlabel("Valores Pixels") #título dos eixos
plt.ylabel("Número de Pixels")

plt.figure('Histograma de um Grão')
plt.subplot(1,3,2)
plt.plot(hist_segmentada_g,color = 'green')
plt.title('Histograma Verde')
plt.xlabel("Valores Pixels") #título dos eixos
plt.ylabel("Número de Pixels")

plt.figure('Histograma de um Grão')
plt.subplot(1,3,3)
plt.plot(hist_segmentada_b)
plt.title('Histograma Azul')
plt.xlabel("Valores Pixels") #título dos eixos
plt.ylabel("Número de Pixels")

plt.show()

print('-'*95)
print('e) Realize a segmentação da imagem utilizando a técnica de k-means. Apresente as imagens\n'
      'obtidas neste processo.')

# Formatação da imagem para uma matriz de dados
pixel_values = img_rgb.reshape((-1, 3))
#muda o formato da matriz, -1 pega todos os pixels e coloca como linha
#3 colunas cada um é referente a um canal

# Conversão para Decimal (para transformar inteiro de 8 bites para decimal)
pixel_values = np.float32(pixel_values)
print('-'*80)
print('Dimensão Matriz: ',pixel_values.shape)
print('-'*80)
########################################################################################################################
# K-means (Processo iterativo)
# Critério de Parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# critérios como medida de precisão em relação ao centroide
# número máximo de iterações que ele vai rodar.
# precisão 0.2

# Número de Grupos (k)
k = 2
# pela imagem achamos que tem K grupos, pode delimitar o número que quizer.
dist, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#pode pedir ele para rodar 10 vezes para encontrar o melhor centroide
#o centroide é chutado aleatoriamente.
#labels= classificação dos pixels eles estão agrupados
print('-'*80)
print('SQ das Distâncias de Cada Ponto ao Centro: ', dist)
print('-'*80)
print('Dimensão labels: ', labels.shape)
print('Valores únicos: ',np.unique(labels))
print('Tipo labels: ', type(labels))
# flatten the labels array
labels = labels.flatten()
print('-'*80)
print('Dimensão flatten labels: ', labels.shape)
print('Tipo labels (f): ', type(labels))
print('-'*80)

# Valores dos labels
val_unicos,contagens = np.unique(labels,return_counts=True)
val_unicos = np.reshape(val_unicos,(len(val_unicos),1))
contagens = np.reshape(contagens,(len(contagens),1))
hist = np.concatenate((val_unicos,contagens),axis=1)
print('Histograma')
print(hist)
print('-'*80)
print('Centroides Decimais')
print(centers)
print('-'*80)
# Conversão dos centroides para valores de interos de 8 digitos
centers = np.uint8(centers)
print('-'*80)
print('Centroides uint8')
print(centers)
print('-'*80)
########################################################################################################################
# Conversão dos pixels para a cor dos centroides
matriz_segmentada = centers[labels]
print('-'*80)
print('Dimensão Matriz Segmentada: ',matriz_segmentada.shape)
print('Matriz Segmentada')
print(matriz_segmentada[0:5,:])
print('-'*80)
########################################################################################################################
# Reformatar a matriz na imagem de formato original
img_segmentada1 = matriz_segmentada.reshape(img_rgb.shape)

# Grupo 1
original_01 = np.copy(img_rgb)
matriz_or_01 = original_01.reshape((-1, 3))
matriz_or_01[labels != 0] = [0, 0, 0]
img_final_01 = matriz_or_01.reshape(img_rgb.shape)

# Grupo 2
original_02 = np.copy(img_rgb)
matriz_or_02 = original_02.reshape((-1, 3))
matriz_or_02[labels != 1] = [0, 0, 0]
img_final_02 = matriz_or_02.reshape(img_rgb.shape)


########################################################################################################################

# Apresentar Imagem
plt.figure('Imagens')
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('ORIGINAL')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(img_segmentada1)
plt.title('ROTULOS')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(img_final_01)
plt.title('Grupo 1')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(img_final_02)
plt.title('Grupo 2')
plt.xticks([])
plt.yticks([])

plt.show()

print('-'*95)
print('f) Realize a segmentação da imagem utilizando a técnica de watershed. Apresente as\n'
      ' imagens obtidas neste processo.')
(L, img_limiar) = cv2.threshold(Cb,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Otsu, pega todas os feijões como um bloco único.
########################################################################################################################
img_dist = ndimage.distance_transform_edt(img_limiar) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
# ndimage.distance_transform_edt: Calcula a distância euclidiana até o zero
# mais próximo (ou seja, pixel de fundo)para cada um dos pixels de primeiro plano.

max_local = peak_local_max(img_dist, indices=False, min_distance=300,
	labels=img_limiar)
# peak_local_max: retorna uma matriz booleana com os picos da imagem
# baseados nas distâncias
# min_distance: Número mínimo de pixels que separam os picos
# https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max
print('-'*50)
print('Número de Picos')
print(np.unique(max_local,return_counts=True))
print('-'*50)

marcadores,n_marcadores = ndimage.label(max_local, structure=np.ones((3, 3)))
# Realiza marcação dos picos
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
# https://en.wikipedia.org/wiki/Connected-component_labeling
print('Análise de conectividade - Marcadores')
print(np.unique(marcadores,return_counts=True))
print('-'*50)

img_ws = watershed(-img_dist, marcadores, mask=img_limiar)
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed
# https://en.wikipedia.org/wiki/Watershed_(image_processing)
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

print('Imagem Segmentada - Watershed')
print(np.unique(img_ws,return_counts=True))
print("Número de Grãos: ", len(np.unique(img_ws)) - 1)
print('-'*50)

img_final = np.copy(img_rgb)
img_final[img_ws != 3] = [0,0,0] # Acessando o grão 3
########################################################################################################################
plt.figure('Watershed')
plt.subplot(2,3,1)
plt.imshow(img_rgb)
plt.xticks([])
plt.yticks([])
plt.title('ORIGINAL')

plt.subplot(2,3,2)
plt.imshow(img_limiar)
plt.xticks([])
plt.yticks([])
plt.title('Limiarização')

plt.subplot(2,3,3)
plt.imshow(img_dist,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Distância')

plt.subplot(2,3,5)
plt.imshow(img_ws,cmap='jet')
plt.xticks([])
plt.yticks([])
plt.title('Grãos de feijão')

plt.subplot(2,3,6)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('SELEÇÃO')

plt.show()

print('-'*95)
print('g) Compare os resultados das três formas de segmentação (limiarização, k-means e watershed\n'
      'e identifique as potencialidades de cada delas.')

# Imprimir a imagem das três.

plt.figure('Imagens')
plt.subplot(1,3,1)
plt.imshow(img_final)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada Wathershed')

plt.subplot(1,3,2)
plt.imshow(img_segmentada)
plt.xticks([])
plt.yticks([])
plt.title('Segmentada OTSU')

plt.subplot(1,3,3)
plt.imshow(img_final_02)
plt.title('Segmentada K-MEANS')
plt.xticks([])
plt.yticks([])

plt.show()
