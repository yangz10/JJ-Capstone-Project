# -*- coding: utf-8 -*-

# Load All Packages
import numpy as np, pandas as pd
import xgboost as xgb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.layers import Input, Dense, Embedding, SpatialDropout1D
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D
from keras.preprocessing import text, sequence

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

np.random.seed(2018)

# Change the Root Path Here
rootPath = '/Users/ValarMorghulis/Johnson_Johnson/FinalCode/'
# read data
class unspsc():

    # Default Path, these paths can be replaced when initializating
    def __init__(self,SapPath = rootPath + 'defaultInput/csvFiles/UNSPSC Full Data Update 2.csv',
                 PrhdaPath= rootPath + 'defaultInput/csvFiles/Prhda.csv',
                 GmdnPath = rootPath + 'defaultInput/csvFiles/GMDN_desp.csv',
                 embedPath = rootPath + 'defaultInput/wordEmbeddingMartix/crawl-300d-2M.vec',
                 weightsPath = rootPath + 'defaultInput/preTrainedWeights/',
                 wordEmPath = rootPath + 'defaultInput/wordEmbeddingMartix/'):

        self.SapPath,self.PrhdaPath  = SapPath, PrhdaPath
        self.EMBEDDING_FILE,self.GmdnPath = embedPath, GmdnPath
        self.weightsPath,self.wordEmPath = weightsPath, wordEmPath

    def dataPre(self):
        """
        Prepare all Data to be used.
        return: two files  - Full data and Y train data
        Unit Test Passed
        """
        # Load All data
        rawData = pd.read_csv(self.SapPath,error_bad_lines=False,encoding='latin1')
        rawPRDHA = pd.read_csv(self.PrhdaPath,error_bad_lines=False,delimiter='\t')
        # Only Select these fields
        columns = ['Breit','Brgew','Hoehe','Laeng','Volum','Zzwerks','Ntgew','Material Description',
                   'Material','Ean11','Gmdnptdefinition','Gmdnptname','Unspsc','Prdha']

        filterData,filterPRDHA = rawData[columns],rawPRDHA[['Prdha','Minor_name','Major_name']]
        # 93 - UNSPSC  73 - UNSPSC
        filterData = filterData.dropna().reset_index(drop=True)
        filterPRDHA['Prdha'] = filterPRDHA['Prdha'].astype('O')

        def prdha_zero(x):
            x, num = str(x), len(str(x))
            if num != 18:
                return '0' * (18 - num) + str(x)
            else:
                return str(x)
        # Fill 18 Digits and Extract First 3 char
        filterData['Prdha'] = filterData['Prdha'].apply(prdha_zero)
        filterPRDHA['Prdha'] = filterPRDHA['Prdha'].apply(prdha_zero)
        # Merge All data
        filterAll = pd.merge(filterData, filterPRDHA, right_on='Prdha', left_on='Prdha', how='inner')
        filterAll['Material_top3'] = filterAll['Material'].apply(lambda x: x[:3])
        y_train = pd.factorize(filterAll['Unspsc'])
        print('Data Preparation Finished')
        return filterAll,y_train

    def Token(self,filterAll,Filed):
        '''
        Word Embedding files here.
        :param Filed: support "material","gmdn","prdha"
        :return:
        Unit Test Passed
        '''

        if Filed == 'material':
            pre_train = filterAll['Material Description'].apply(str).str.lower()
            pre_test = filterAll['Material Description'].apply(str).str.lower()
            max_features = 4000 # If you want to change, check the matrix length af first
        if Filed == 'gmdn':
            rawGMDN = pd.read_csv(self.GmdnPath, error_bad_lines=False, encoding='utf8')
            pre_train = rawGMDN['Short Desp'].apply(str).str.lower()
            pre_test = filterAll['Gmdnptname'].apply(str).str.lower()
            max_features = 700
        if Filed == 'prdha':
            rawPRDHA = pd.read_csv(self.PrhdaPath, error_bad_lines=False, delimiter='\t')
            pre_train = rawPRDHA['Minor_name'].apply(str).str.lower()
            pre_test = filterAll['Minor_name'].apply(str).str.lower()
            max_features = 1000

        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(pre_train))
        x_train = unspsc.tokenSide(self,tokenizer,pre_train)
        x_test = unspsc.tokenSide(self,tokenizer,pre_test)
        return x_train, x_test, tokenizer, max_features

    def TokenInput(self,dataset,Filed):
        '''
        Word Embedding files here.
        :param Filed: support "material","gmdn","prdha"
        :return:
        Unit Test Passed
        '''

        if Filed == 'material':
            pre_train = filterAll['Material Description'].apply(str).str.lower()
            pre_test = dataset['Material Description'].apply(str).str.lower()
            max_features = 4000
        if Filed == 'gmdn':
            rawGMDN = pd.read_csv(self.GmdnPath, error_bad_lines=False, encoding='utf8')
            pre_train = rawGMDN['Short Desp'].apply(str).str.lower()
            pre_test = dataset['Gmdnptname'].apply(str).str.lower()
            max_features = 700
        if Filed == 'prdha':
            rawPRDHA = pd.read_csv(self.PrhdaPath, error_bad_lines=False, delimiter='\t')
            pre_train = rawPRDHA['Minor_name'].apply(str).str.lower()
            pre_test = dataset['Minor_name'].apply(str).str.lower()
            max_features = 1000

        tokenizer = text.Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(pre_train))
        x_train = unspsc.tokenSide(self,tokenizer,pre_train)
        x_test = unspsc.tokenSide(self,tokenizer,pre_test)
        return x_train, x_test, tokenizer, max_features

    def tokenSide(self,tokenizer,text,maxlen=100):
        '''
        :param tokenizer: Pre-Trained tokenizer
        :param text: Input Text
        :param maxlen: Maximum Feature Len
        :return: Cleaned text
        Unit Test passed
        '''
        textStep = tokenizer.texts_to_sequences(text)
        testFinal = sequence.pad_sequences(textStep, maxlen=maxlen)
        return testFinal

    def wordEmbedding(self,tokenizer,max_features,Filed,embed_size = 300,PreTrained=True):
        '''
        Genrate Word Embedding Matrix
        :param tokenizer:
        :param max_features:
        :param Filed:
        :param embed_size:
        :param PreTrained: Whether It is Pretained
        :return: Word Embedding matrx
        Unit Test passed
        '''
        if PreTrained :
            if Filed == 'material':
                embedFile = self.wordEmPath + 'material_em.npy'
            if Filed == 'gmdn':
                embedFile = self.wordEmPath + 'gmdn_em.npy'
            if Filed == 'prdha':
                embedFile = self.wordEmPath + 'prdha_em.npy'
            embedding_matrix = np.load(embedFile)
        else :
            def get_coefs(word, *arr):
                return word, np.asarray(arr, dtype='float32')

            embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(self.EMBEDDING_FILE,encoding='utf-8'))

            word_index = tokenizer.word_index
            nb_words = min(max_features, len(word_index))
            embedding_matrix = np.zeros((nb_words, embed_size))
            for word, i in word_index.items():
                if i >= max_features: continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def modelPre(self,Filed,x_test,model):
        '''
        modelPre mainly loads all pre-trained paramters to accelerate final results presentations
        :param Filed: Whether to use gmdn, prdha or material descriptions
        :param x_test:  Dataset used to predict
        :param model: RNN final models used for obtain the final hidden layers
        :return: Predictions
        '''
        if Filed == 'material':
            weight = self.weightsPath + 'MaterialDesp.h5'
        if Filed == 'gmdn':
            weight = self.weightsPath + 'GMDN_Desp.h5'
        if Filed == 'prdha':
            weight = self.weightsPath + 'prdha.h5'
        model.pop()
        pred = model.predict(x_test, batch_size=512, verbose=1)
        return pred

    def modelNetTrain(self,Filed,x_test,epochs,model,y_train):
        '''
        modelNetTrain using RNN to get hidden layers.
        :param Filed: Whether to use gmdn, prdha or material descriptions
        :param x_test: Dataset used to predict
        :param epochs: Number of epoches
        :param model: RNN final models used for obtain the final hidden layers
        :param y_train: Labels for training
        :return: Predictions
        '''
        batch_size = 512
        X_tra, X_val, y_tra, y_val = train_test_split(x_test, y_train[0], train_size=0.8, random_state=125)
        hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs,
                         validation_data=(X_val, y_val), verbose=1)
        newPath = self.weightsPath + 'New_' + Filed + '.h5'
        model.save_weights(newPath)
        model.pop()
        pred = model.predict(x_test, batch_size=512, verbose=1) # if need to change
        return pred

    def modelTrain(self,filterAll,y_train,Filed,Pre_trained=True,Summary=False,epochs = 20,Input=True):
        '''
        modelTrain is a unit train model which will train each text field using RNN.
        :param filterAll: The orignal dataset - can be regarded as a database
        :param y_train: labels
        :param Filed: Whether to use gmdn, prdha or material descriptions
        :param Pre_trained: Whether to load pre-trained matrix
        :param Summary: Whether show the model layers
        :param epochs: Number of epoches
        :return: Predictions (Hidden Layers and word embeddings)
        '''
        if Input:
            x_train, x_test, tokenizer, max_features = unspsc.Token(self, filterAll, Filed=Filed)
        else :
            x_train, x_test, tokenizer, max_features = unspsc.Token(self, filterAll, Filed=Filed)

        embedding_matrix = unspsc.wordEmbedding(self,tokenizer=tokenizer, max_features=max_features,
                                                Filed=Filed,embed_size=300, PreTrained=True)
        embed_size =300

        model = Sequential()
        embed = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix])
        model.add(embed)
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(GRU(80, return_sequences=True)))
        model.add(GlobalAveragePooling1D())
        if Filed == 'prdha':
            model.add(Dense(200, activation="sigmoid"))
        else :
            model.add(Dense(100, activation="sigmoid"))
        if Input:
            model.add(Dense(y_train[1], activation="softmax"))
        else :
            model.add(Dense(len(y_train[1]), activation="softmax"))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        if Summary:
            model.summary()

        if Pre_trained :
            pred = unspsc.modelPre(self,Filed,x_test,model)
        else :
            pred = unspsc.modelNetTrain(self,Filed,x_test,epochs,model,y_train)
        return pred

    def stagingFeature(self,filterAll,y_train,Pre_trained=True,Summary=False,epochs=20,loadCSV=True,Input=True):
        '''
        stagingFeature mainly deal with all word embedding output. It calls all functions to transform previous
        word embedding output.
        :param Pre_trained: Whether load pre-trained data to save time
        :param Summary: Whether show the neutral network layers summary
        :param epochs: How many epoches want to use
        :return: Word Embedding Layers seperately
        '''
        if loadCSV:
            material_path = self.wordEmPath + 'material_output.npy'
            gmdn_path = self.wordEmPath + 'gmdn_output.npy'
            prdha_path = self.wordEmPath + 'prdha_output.npy'
            gmdn_output = np.load(material_path)
            material_output = np.load(gmdn_path)
            prdha_output = np.load(prdha_path)
        else :
            gmdn_output = unspsc.modelTrain(self,filterAll,y_train,'gmdn',Pre_trained=Pre_trained,Summary=Summary,epochs = epochs,Input=Input)
            material_output = unspsc.modelTrain(self,filterAll,y_train, 'material', Pre_trained=Pre_trained, Summary=Summary, epochs=epochs,Input=Input)
            prdha_output = unspsc.modelTrain(self,filterAll,y_train, 'prdha', Pre_trained=Pre_trained, Summary=Summary, epochs=epochs,Input=Input)
        return gmdn_output,material_output,prdha_output

    def FeatureMerge(self,gmdn_output,material_output,prdha_output,filterAll):
        '''
        FeatureMerge function is preparing dataset for later training.
        :param gmdn_output: GMDN word embedding matrix
        :param material_output: Material word embedding matrix
        :param prdha_output: Prdha word embedding matrix
        :param filterAll: DataSet with all fields.
        :return: Merged Data Matrix
        '''
        gmdn_output,material_output,prdha_output = pd.DataFrame(gmdn_output),pd.DataFrame(material_output),pd.DataFrame(prdha_output)
        embFeatures = pd.concat([gmdn_output,material_output,prdha_output], axis=1)
        location = pd.get_dummies(filterAll['Zzwerks'], prefix='Location')
        featuresReady = pd.concat([filterAll.iloc[:, :5], location, embFeatures], axis=1)
        return featuresReady

    def finalPrediction(self,featuresReady,y_train,num_round=50,preTrained=True,InputTest=True):
        '''
        finalPrediction will give final prediction for a given matrix.
        :param featuresReady: This is a matrix with all merged data like word embeddings, locations and product dimensions.
        :param y_train: If preTrained is true (want to train the model), we have to give certain labels.
        :param num_round: How many rounds to train the data.
        :param preTrained: If we will load pre-trained model to save time, this should be true.
        :param InputTest: Whether predict all values based on new input dataset.
        :return: Final predictions
        '''
        if preTrained:
            bst = xgb.Booster(model_file=self.wordEmPath+'xgboost.model')
        else :
            featuresReady_matrix = np.matrix(featuresReady)
            X_train, X_test, y_train_, y_test_ = train_test_split(featuresReady_matrix, y_train[0], test_size=0.4,
                                                                  random_state=404)
            # factorlized all input
            xg_train = xgb.DMatrix(X_train, label=y_train_)
            xg_test = xgb.DMatrix(X_test, label=y_test_)
            # input parameters and models
            param = {'objective': 'multi:softmax', 'eta': 0.2, 'silent': 0,
                     'num_class': len(y_train[1]),
                     'gamma': 0.2,
                     }
            watchlist = [(xg_train, 'train'), (xg_test, 'test')]
            num_round = num_round
            bst = xgb.train(param, xg_train, num_round, watchlist)
            bst.save_model('new_xgboost.model')
        # get prediction
        if InputTest :
            data = np.matrix(featuresReady)
            data = xgb.DMatrix(data)
            pred = bst.predict(data)
        else :
            featuresReady_matrix = np.matrix(featuresReady)
            X_train, X_test, y_train_, y_test_ = train_test_split(featuresReady_matrix, y_train[0], test_size=0.4,
                                                                  random_state=404)
            # factorlized all input
            xg_train = xgb.DMatrix(X_train, label=y_train_)
            xg_test = xgb.DMatrix(X_test, label=y_test_)
            pred = bst.predict(xg_test)
            error_rate = np.sum(pred != y_test_) / y_test_.shape[0]
            print('Test error using softmax = {}'.format(error_rate))
            print('Accuracy is {}'.format(accuracy_score(y_test_, pred)))
        return pred

    def dataInput(self,dataSet):
        # same transformations

        filterAll, y_train_ = unspsc.dataPre(self)
        y_train =[1,73]
        gmdn_output, material_output, prdha_output = unspsc.stagingFeature(self,dataSet, y_train, Pre_trained=True,Summary=False,
                                                                           epochs=20,loadCSV=False,Input=True)
        featuresReady = unspsc.FeatureMerge(self,gmdn_output, material_output, prdha_output, dataSet)
        finalPrediction = unspsc.finalPrediction(self,featuresReady, y_train, num_round=50,preTrained=True,InputTest=True)
        return (y_train_[1][int(finalPrediction)])


if __name__ == "__main__":
    print('Demo')
    # model = unspsc()
    # filterAll, y_train = model.dataPre()

    # gmdn_output, material_output, prdha_output = model.stagingFeature(filterAll,y_train,Pre_trained=True,Summary=False,epochs=20)
    # featuresReady = model.FeatureMerge(gmdn_output,material_output,prdha_output,filterAll)
    # finalPrediction = model.finalPrediction(featuresReady, y_train, num_round=50)
