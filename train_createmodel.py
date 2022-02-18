import os
import pywt
import mahotas
import numpy as np
from scipy.signal import correlate2d
from scipy import ndimage
from scipy import stats
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
import sklearn.metrics as metrics
from time import time as tick
import joblib
import matplotlib.pyplot

import qimage2ndarray as qn

from test_segment import *


#Line detector generico
def linedetectorWl(img, mask, borda, vasos, outline=1000):
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

        if vasos[i_orig,j_orig]>0:
            label=1
        labels.append(label)


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
            image[i-7:i+8, j-7:j+8] = k #por a img igual
        
    return features, labels


#Extração de features Line detector
def linedetector_train(img, mask, borda, vasos, i, outline=1000, proportion='random'):
    print("Extracting image %d features" %(i+1))
    image,_ = add_outlines(img, outline*2)

    #[0 15 30 45]
    angles=np.arange(0,60,15)*np.pi/180
    declive=np.tan(angles)
    features=[]
    labels=[]
    #Direções dos angulos: [0,15,30,60,75,90,-75-60,-45,-30,-15]
    ortogonais=[6,6,9,9,9,0,0,0,3,3,3,6] #[90,90,-45,-45,-45,0,0,0,45,45,45,90]
    

    
    if proportion=='random':
        linhas, colunas = np.nonzero(mask>0)
        ii, jj= shuffle(linhas, colunas, n_samples=1000, random_state=0)
    else:
        background=mask-vasos
        p_vasos=int(proportion[0]*1000) #numero de pontos correspondentes ao vasos
        p_background=int(proportion[1]*1000)
        
        linhas, colunas = np.nonzero(vasos>0)  #indices correspondentes aos vasos
        ii_v, jj_v= shuffle(linhas, colunas, n_samples=p_vasos, random_state=0) #escolher p_vasos desses indices, aleatoriamente
        
        linhas, colunas = np.nonzero(background>0)
        ii_b, jj_b= shuffle(linhas, colunas, n_samples=p_background, random_state=0)
        
        #https://www.geeksforgeeks.org/python-ways-to-concatenate-two-lists/
        ii=[*ii_v, *ii_b] #juntar as linhas de vasos e background, num total de 1000 indices
        jj=[*jj_v, *jj_b] #juntar as colunas correspondentes, pela mesma ordem
        
        labels += p_vasos*[1] + p_background*[0] 
        ii, jj, labels = shuffle(ii, jj, labels, random_state=0) #fazer o shuffle das matrizes, de forma igual, mantendo as correspondencias entre o i, j e label de cada pixel
                                                 #para nao ficar sempre com todos os pixeis de vasos juntos e depois os de background; para nao comprometer o treino e construção do modelo
                                                 
    for i_orig,j_orig in zip(ii,jj):
        label=0
        yy=[]
        xx=[]
        i=i_orig + outline
        j=j_orig + outline
        
        if proportion=='random':  #so precisamos de fazer a matriz das labels, se o proporção for aleatoria
            if vasos[i_orig,j_orig]>0:
                label=1
            labels.append(label)

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
            image[i-7:i+8, j-7:j+8] = k #por a img igual
        
    return features, labels
    

def construir_treino (train, labels_train, proporcao='random'):
    print('Extracting features')	
    features=[]
    y_t=[]
    n_images= int(0.6*len(train))
    training, labels_training= shuffle(train, labels_train, n_samples=n_images, random_state=0)
    masks= []
    bordas = []

    for i in range (len(training)):
        name =training[i]
        print(str(i+1) + ':', name)
        image_orig=readImage(name)
        image=inverter(image_orig)
        mask,borda = FOV_border(image_orig)
        
        masks.append(mask)
        bordas.append(borda)

        name_vaso= labels_training[i]
        vasos=mahotas.imread(name_vaso)
        
        ft, labels_train= linedetector_train(image, mask, borda, vasos, i, proportion=proporcao)
        nft=normalizacao(ft)
        features.append(nft)
        y_t.append(labels_train)
        
    x_train=np.asarray(features) #array 3D - 20 imagens/matrizes - 1000 linhas=pixeis, cada uma com 3 colunas=features
    x_train=np.reshape(x_train, (-1, 3))

    y_train=np.asarray(y_t)
    y_train=np.reshape(y_train, (-1, ))
    

    return x_train, y_train, training, labels_training, masks, bordas

    
def build_model(Params):
    print('Building model')
    if Params[0] == 'Support Vector Machine':
        if Params[1]=='rbf':
            model = SVC(kernel=Params[1], C=Params[2], gamma=Params[3], random_state=0) 
            print('rbf')
        else:
            model = SVC(kernel=Params[1], C=Params[2], random_state=0)
            print('linear')

    elif Params[0] == 'Random Decision Forests':
        model = RandomForestClassifier(n_estimators=Params[1], max_depth=Params[2], min_samples_split=Params[3], max_features=Params[4], max_leaf_nodes=Params[5], random_state=0)
        print('Forests')

    else:
        model = ExtraTreesClassifier(n_estimators=Params[1], max_depth=Params[2], min_samples_split=Params[3], max_features=Params[4], max_leaf_nodes=Params[5], random_state=0)
        print('Trees')
    return model
    
    
    
def test_train(train, labels_train, masks, bordas, model, model_name):
    print('Training images segmentation')
    try:  
       os.mkdir(os.getcwd()+'/Segmentation_Results/Train/'+ str(model_name))  
    except OSError as error:  
       print(error)

    path=os.getcwd()+'/Segmentation_Results/Train/'+ str(model_name)

    ACCs=[]
    AUCs=[]
    Precisions=[]
    Recalls=[]
    DICEs=[]

    for i in range (len(train)):
        name = train[i]

        img_name=os.path.splitext(os.path.basename(name))[0]

        print('Train Image ' + str(i+1) + ': ' + str(img_name))

        image_original=readImage(name)
        image=inverter(image_original)
        mask=masks[i]
        borda=bordas[i]

        name_vaso=labels_train[i]
        vasos=mahotas.imread(name_vaso)

        ft_train, labels_t= linedetectorWl(image, mask, borda, vasos)
        nft_train=normalizacao(ft_train)
        x_train_totest=np.asarray(nft_train)
        y_train_totest=np.asarray(labels_t)

        y_pred_train= model.predict(x_train_totest)
        img_pred_train= imagem_segmentada(image_original, y_pred_train, mask)

        matplotlib.pyplot.imsave(path+'/'+str(img_name)+ '_train.gif', img_pred_train, cmap='gray')

        #qim=qn.array2qimage(img_pred_train)
        #w=qim.save(path+'/'+str(img_name)+ '_train.tiff', format ='TIFF') 
        print('Training image %s saved' %(i+1))
        
        #MTERICAS
        acc = metrics.accuracy_score(y_train_totest, y_pred_train)
        ACCs.append(acc)
        
        AUC=metrics.roc_auc_score(y_train_totest, y_pred_train)
        AUCs.append(AUC)
        
        precision=metrics.precision_score(y_train_totest, y_pred_train)
        Precisions.append(precision)
        
        recall=metrics.recall_score(y_train_totest, y_pred_train)
        Recalls.append(recall)
        
        f1=metrics.f1_score(y_train_totest, y_pred_train)
        DICEs.append(f1)

    mean_acc=sum(ACCs)/len(ACCs)
    mean_auc=sum(AUCs)/len(AUCs)
    mean_precision=sum(Precisions)/len(Precisions)
    mean_recall=sum(Recalls)/len(Recalls)
    mean_DICE=sum(DICEs)/len(DICEs)

    metrics_train={'ACC':mean_acc, 'AUC':mean_auc, 'Precision':mean_precision, 
    'Recall': mean_recall, 'DICE':mean_DICE}

    print("Mean train ACC score: {0:.2f} %".format(100 * mean_acc))
    print("Mean train AUC score: {0:.2f} %".format(100 * mean_auc))
    print("Mean train Precison score: {0:.2f} %".format(100 * mean_precision))
    print("Mean train Recall score: {0:.2f} %".format(100 * mean_recall))
    print("Mean train DICE score: {0:.2f} %".format(100 * mean_DICE))

    return metrics_train

        
def test_test(test, labels_test, model, model_name):
    print('Test images segmentation')
    try:  
       os.mkdir(os.getcwd()+'/Segmentation_Results/Test/'+ str(model_name))  
    except OSError as error:  
        print(error)

    path=os.getcwd()+'/Segmentation_Results/Test/'+ str(model_name)

    ACCs=[]
    AUCs=[]
    Precisions=[]
    Recalls=[]
    DICEs=[]

    for i in range (len(test)):
        name = test[i]

        img_name=os.path.splitext(os.path.basename(name))[0]

        print('Test Image ' + str(i+1) + ': ' + str(img_name))
        image_original=readImage(name)
        image=inverter(image_original)
        mask, borda = FOV_border(image_original)

        name_vaso=labels_test[i]
        vasos=mahotas.imread(name_vaso)

        ft_test, labels_t= linedetectorWl(image, mask, borda, vasos)
        nft_test=normalizacao(ft_test)
        x_test=np.asarray(nft_test)
        y_test=np.asarray(labels_t)

        y_pred= model.predict(x_test)
        img_pred_test= imagem_segmentada(image_original, y_pred, mask)
        

        matplotlib.pyplot.imsave(path+'/'+str(img_name)+ '_test.gif', img_pred_test, cmap='gray')

        #qim=qn.array2qimage(img_pred_test)
        #w=qim.save(path+'/'+str(img_name)+ '_test.tiff', format ='TIFF') 
        #print('w:', w)
        print('Test image %s saved' %(i+1))

        #MTERICAS
        acc = metrics.accuracy_score(y_test, y_pred)
        ACCs.append(acc)
        
        AUC=metrics.roc_auc_score(y_test, y_pred)
        AUCs.append(AUC)
        
        precision=metrics.precision_score(y_test, y_pred)
        Precisions.append(precision)
        
        recall=metrics.recall_score(y_test, y_pred)
        Recalls.append(recall)
        
        f1=metrics.f1_score(y_test, y_pred)
        DICEs.append(f1)

        print('Test image %s saved' %(i+1))

    mean_acc=sum(ACCs)/len(ACCs)
    mean_auc=sum(AUCs)/len(AUCs)
    mean_precision=sum(Precisions)/len(Precisions)
    mean_recall=sum(Recalls)/len(Recalls)
    mean_DICE=sum(DICEs)/len(DICEs)

    metrics_test={'ACC':mean_acc, 'AUC':mean_auc, 'Precision':mean_precision, 
    'Recall': mean_recall, 'DICE':mean_DICE}

    print("Mean test ACC score: {0:.2f} %".format(100 * mean_acc))
    print("Mean test AUC score: {0:.2f} %".format(100 * mean_auc))
    print("Mean test Precison score: {0:.2f} %".format(100 * mean_precision))
    print("Mean test Recall score: {0:.2f} %".format(100 * mean_recall))
    print("Mean test DICE score: {0:.2f} %".format(100 * mean_DICE))

    return metrics_test



	  	
def train_model(Params, files, files_gt, name):
    model=build_model(Params)
    x_train, y_train, training, labels_training, masks, bordas= construir_treino (files, files_gt, proporcao=[0.1,0.9])
    print('Training')
    tin = tick()
    model.fit(x_train, y_train)
    tout = tick()
    time=tout - tin
    print('Training time: {:.3f} s'.format(time))
    metrics_train= test_train(training, labels_training, masks, bordas, model, name)

    test=[]
    labels_test=[]
    for i in range (len(files)):
        x= files[i]
        if x not in training:
            test.append(x)
            labels_test.append(files_gt[i])
    metrics_test= test_test(test, labels_test, model, name)

    results={"Model":model, "Time":time, "TrainMetrics": metrics_train, "TestMetrics":metrics_test}

    return results
	
	
