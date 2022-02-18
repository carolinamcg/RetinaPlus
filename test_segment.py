import os
import pywt
import mahotas
import numpy as np
from scipy.signal import correlate2d
from scipy import ndimage
from scipy import stats
#from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from time import time as tick

def readImage(path, canal=1):
    image=mahotas.imread(path)
    image2=image[:,:,canal] #RGB
    return image2

def inverter (image_original):
    a=-1
    #y=mx -> o y do centro passa a ser: yc = -127,5, ou seja centro = (127,5 , -127,5) -> deixa de estar centrado
    b=2*127.5 #para manter o centro da reta em 127,5 (o centro da gama de cinzas)
    image=np.array(image_original)

    for i in range (image.shape[0]):
        for j in range (image.shape[1]):
            if ((a*image[i,j])+b)<0:
                image[i,j]=0
            elif (a*image[i,j])+b>255:
                image[i,j]=255
            else:
                image[i,j]=(a*image[i,j])+b
    return image

def add_outlines(image, kernel_size):
    borda = kernel_size//2 #neste caso, como kernel_size=3, a=1   #espessura da moldura
    rows=image.shape[0]
    cols=image.shape[1]
    new = np.zeros((rows+borda*2, cols+borda*2), dtype=np.float32) #temos que adicionar bordas, para quando o kernel se situar nos pontos extremos da imagem
                                #temos que adicionar 2*borda: uma linha a cima e outra a baixo da imagem ( o mesmo para as colunas)
    new[borda:-borda, borda:-borda] = image #coloca a nossa imagem na submatriz central de new, de modo a ficar com as bordas a 0
                                # primeiras e ultimas "borda" linhas ficam a 0 (o mesmo para as colunas)
    return (new, borda)

def remove_outlines(image,borda):
    rows=image.shape[0]
    cols=image.shape[1]
    filt = np.zeros((rows-borda*2, cols-borda*2), dtype=np.float32) #temos que retirar bordas
    filt= image[borda:-borda, borda:-borda]  #coloca a nossa imagem na submatriz central de filt
    return (filt)

def opening(img):
    kernel=mahotas.disk(20) #retorna um kernel circular 2D de raio 20
    eros=mahotas.erode(img.astype(np.uint8),kernel)
    dil=mahotas.dilate(eros.astype(np.uint8),kernel)
    return dil
    
def closing(img):
    kernel=mahotas.disk(8)
    dil=mahotas.dilate(img.astype(np.uint8),kernel)
    eros=mahotas.erode(dil.astype(np.uint8),kernel)
    return eros

def sobel(img, A=1):
    kernel1=np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel2=np.transpose(kernel1)
    i1=correlate2d(img, kernel1, 'same')
    i2=correlate2d(img, kernel2, 'same')
    img_filt=np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_filt[i,j]=np.sqrt(i1[i,j]**2+i2[i,j]**2)
    img_filt2=A*img_filt
    return img_filt2

def elementos_ligados(img):
    labeled, nr_objects = mahotas.label(img) 
    tamanhos=mahotas.labeled.labeled_size(labeled) 
    tamanhos_ord=np.sort(tamanhos)
    border_int = np.nonzero(tamanhos == tamanhos_ord[-2])[0][0]
    background = np.nonzero(tamanhos == tamanhos_ord[-1])[0][0]
    binary=np.zeros(img.shape)
    binary[labeled == border_int]= 255 
    binary[labeled == background]= 0
    return binary

def th(img): #A função utso calcula automaticamente o melhor threshold para binarizar a imagem
    T_otsu = mahotas.otsu(img.astype(np.uint8)) #img should be of an unsigned integer type.
    binarized_image = (img > T_otsu)
    img_bin=np.zeros(img.shape)
    img_bin[binarized_image>0]=255
    return img_bin

def FOV_border(image):
    print("A identificação a FOV e borda!")
    img_padded,_ = add_outlines(image, 60)
    img_closed= closing(img_padded)
    img_op=opening(img_closed)
    img= remove_outlines(img_op, 30)
    ISOBEL = sobel(img)
    binary1=th(ISOBEL)
    img_ligada= elementos_ligados(binary1)
    kernel=mahotas.disk(3)
    IDILAT = mahotas.dilate(img_ligada.astype(np.uint8),kernel)
    I_SUB = IDILAT-img_ligada

    #-  Após ter imagem binarizada para se conseguir definir apenas o contorno do olho tem de se fazer labeling,
        #recorrendo a funçoes do mahotas
    #- Os dois maiores elementos ligados são o fundo e a borda exterior 
    # ----> a borda interior é o terceiro maior elemento ligado
    labeled, nr_objects = mahotas.label(I_SUB) #labeled = array do tamanho de img_sub, onde, a cada pixel ele atribui o numero do label correspondente (0,1,...)
    tamanhos=mahotas.labeled.labeled_size(labeled) #da o tamanho de cada objecto, por ordem de labels ([label 0 label 1 ...])
    tamanhos_ord=np.sort(tamanhos)
    border_int = np.nonzero(tamanhos == tamanhos_ord[-3])[0][0] #nonzero da um array com a posição onde a condicao se verifica
    #esta posicao corresponde ao label X, logo, todos os pixeis da imagem onde labeled == X correspondem à bord

    #- Após seleçao da borda pode-se criar uma imagem binaria com a borda
    binary2=np.zeros(I_SUB.shape)
    binary2[labeled == border_int]= 255 

    #- para criar a máscara é necessário preencher a borda
    mask= ndimage.binary_fill_holes(binary2)  #fil holes para preencher buracos
    mask=mask*255
    
    return mask, binary2

def linedetector(img, mask, borda, vasos=None, outline=1000):
    print("A extrair as features!")
    image,_ = add_outlines(img, outline*2)

    #[0 15 30 45]
    angles=np.arange(0,60,15)*np.pi/180
    declive=np.tan(angles)
    features=[]
    labels=[]
    #Direções dos angulos: [0,15,30,60,75,90,-75-60,-45,-30,-15]
    ortogonais=[6,6,9,9,9,0,0,0,3,3,3,6] #[90,90,-45,-45,-45,0,0,0,45,45,45,90]
    
    ii, jj = np.nonzero(mask>0)
    for i_orig,j_orig in zip(ii,jj):
        label=0
        yy=[]
        xx=[]
        i=i_orig + outline
        j=j_orig + outline

        x=np.arange(j-7, j+8).astype(int)

        xx.extend([x for i in range(4)]) # angulos 0, 15, 30, 45

        for m in declive:
            y=np.around(np.multiply(m,x)+i-m*j).astype(int)
            yy.append(y)
            
        y=np.arange(i-7, i+8).astype(int)

        yy.extend([y for i in range(3)]) # angulos 60, 75, 90 e os seus negativos
        for m in declive:
            x=np.around(np.multiply(m,y)+j-m*i).astype(int)
            xx.insert(4, x) #colocar por ordem

        xx.pop(4) #remover repetição de 45º


        #Fazer para os negativos
        for a in range (len(xx)-2, 0, -1):
            x=xx[a]
            xinv=x[::-1]
            xx.append(xinv)
            yy.append(yy[a])

        
        if borda[i_orig, j_orig] >0: #ponto de borda
            k=img[i_orig-7:i_orig+8, j_orig-7:j_orig+8]
            fov = mask[i_orig-7:i_orig+8, j_orig-7:j_orig+8]
            kernel=k
            med=kernel[fov>0].mean()
            kernel[fov==0] = np.round(med)  
            image[i-7:i+8, j-7:j+8]=kernel
            #image[i+1993:i+1008, j+1993:j+1008]=kernel #i na image = i na img + borda, com borda=1000
            
            media=0
            L = 0
            direcao=None
            
            for z in range(len(xx)):
                x=xx[z]
                y=yy[z]
                line=image[y,x]
                media = line.mean()
                if media >= L:
                    L = media
                    direcao = z
            
        else:    
            media=0
            L = 0
            direcao=None
            for z in range(len(xx)):
                x=xx[z]
                y=yy[z]
                line=image[y,x]
                media = line.mean()
                if media >= L:
                    L = media
                    direcao = z
           # if i==1300 and j==1301:
            #    print('Pixel central:', image[i,j])
            #    print('Intensidade média, Direção:',L, direcao)
             #   print('coordenadas linhas:', xx[direcao], yy[direcao])
              #  print('Linha de direção:',image[yy[direcao],xx[direcao]])
               # print(image[i-7:i+8, j-7:j+8])


        N=image[i-7:i+8, j-7:j+8].mean()
        S= L -N


        #So
        index_prep=ortogonais[direcao]
        xo=xx[index_prep]
        yo=yy[index_prep]
        orto=image[[yo[6:9]], [xo[6:9]]]
        Lo=orto.mean()
        So=Lo - N

        I=image[i,j]

        features.append([S, So, I])
        
        if  borda[i_orig, j_orig] >0:
            image[i-7:i+8, j-7:j+8] = k 
        
    return features

def normalizacao(features):
    print("A normalizar!")
    features_norm=np.zeros_like(features)
    for col in range (len(features[0])):
        mean= sum([item[col] for item in features])/len(features)
        desvio= np.std([item[col] for item in features])
        for line in range (len(features)):
            value = (features[line][col]-mean)/desvio
            features_norm[line][col]=value
    return features_norm

def imagem_segmentada(image, labels, mask):
    print("A construir a imagem segmentada!")
    new_image=np.zeros(image.shape)
    #print(len(features[:,2]))
    new_image[np.nonzero(mask>0)]=labels
    new_image[new_image<0]=0
    new_image[new_image>255]=255
    return new_image

def segmentar(path, model):
    name =path
    image_original=readImage(name)
    if path.endswith("03_test.tif"): #imagem com pontos de borda muito proximos da ultima coluna da image, nao possuem 7 pixeis entre esta e eles
        print("Imagem com dimensões diferentes!")
        image_original=np.insert(image_original, [564,564], 0, axis=1)
    image=inverter(image_original)
    mask,borda = FOV_border(image_original)
    ft_test= linedetector(image, mask, borda)
    nft_test=normalizacao(ft_test)
    x_test=np.asarray(nft_test)

    y_pred = model.predict(x_test)
    img_segmentada = imagem_segmentada(image_original, y_pred, mask)
    img_segmentada[img_segmentada>0]=255

    return img_segmentada