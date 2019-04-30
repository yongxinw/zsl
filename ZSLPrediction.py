from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

class ZSLPrediction(Object)
    
    def __init__(self, train_cls, test_cls, word2vec='spacy'):

        self.train_cls = train_cls #num_train_cls
        self.test_cls = test_cls   #num_test_cls
        #Calculate word embeddings of train and test class

        self.train_word_embeddings = self._to_embeddings(train_cls, word2vec) #num_train_cls x embedding dim
        self.test_word_embeddings = self._to_embeddings(test_cls, word2vec) #num_test_cls x embedding dim

    def _to_embeddings(self, list_of_string, word2vec):
        if word2vec == 'spacy':
            import os 
            os.system('python -m spacy download en_core_web_sm')
            import spacy
            nlp = spacy.load("en_core_web_sm")

            list_of_embeddings = []
            for each_word in list_of_string:
                embedding = nlp(each_word).vector() 
                list_of_embeddings.append(embedding)
            return np.vstack(list_of_embeddings)
        elif word2vec == 'gensim':
            raise NotImplementedError 
        

    def num_train_cls(self):
        return len(self.train_cls)

    def num_test_cls(self):
        return len(self.test_cls)

    def conse_wordembedding_predict(self, score, topN=None):

        """
        Input:
            score ndarray : 
                The score for each seen class
                siee: N x number of seen class
            
        Output: 
            predictions:
                The index of class in test set
                size: N x 1 
        """
        assert self.train_word_embeddings is not None, "Need word2vec first"
        assert self.test_word_embeddings is not None, "Need word2vec first"

        num_samples = score.shape[0]
        if topN is not None:
            belowN_index = np.argsort(score, axis = 1)[::-1][:,topN:] #sort score 
            for i in range(num_samples):
                score[i,belowN_index] = 0
        convex_vector = score.dot(self.train_word_embeddings) #N x embedding dim

        def _cos_sim_cls(v):
            return self.cos_sim(v.reshape(1,-1), self.test_word_embeddings)

        similarity = np.apply_along_axis(_cos_sim_cls, 1, convex_vector)
        prediction = np.argmax(similarity, axis = 1)

        return prediction


    def cos_sim(self, v1, v2):
        """
        v1 is 1 x N or M x N
        v2 is M x N
        """
        dot_prod = np.sum(v1 * v2, axis = 1)
        v1_magnitude = np.sqrt(np.sum(v1 * v1, axis = 1)) #a length
        v2_magnitude = np.sqrt(np.sum(v2 * v2, axis = 1)) #b length
        return dot_prod/(v1_magnitude * v2_magnitude)

    def construct_nn(self, all_test_features, all_test_label, k = 5, metric = 'cosine',\
                        sample_num = 5, sample_ratio = None):
        """
            Input:
                all_test_features  
                    size: N x feature_size
                all_test_label
                    size: N x 1
            
            KNN related:
                k: 
                    number of nearest neighbors
                metric:
                    metric to use, can be 'euclidean', 'cosine', etc
            
            Sampling related:
                sample_num
                    number of samples
                sample_ratio
                    number of sample by ratio of corresponding class

        """
        anchors = self._get_anchors(all_test_features, all_test_label, sample_num, sample_ratio)

        self.nn = NearestNeighbors(n_neighbors = k, metric = metric)
        features, label = anchors
        self.nn.fit(features)
        self.anchors_label = label

    def nn_predict(self, features):
        """
        Input: 
            features: 
                visual feature to predict
                size: N x feature_size 
        """
        dists, neighbors = self.nn.kneighbors(features)
        return self.anchors_label[neighbors]

    def tSNE_visualization(self, features, labels, mode='test'):
        """
        Input:
            features:
                visual feature to show
                size: N x feature_size
            labels:
                label for each sample
            mode:
                if test, use test class
                if train, use train class
        """
        class_to_use = None
        if mode == 'test':
            class_to_use = self.test_cls
        elif mode == 'train':
            class_to_use = self.train_cls
        num_class = len(class_to_use)

        transformed_features = TSNE(n_components=2).fit_transform(features)

        colors = label.astype(np.float)/num_class

        fig, ax = plt.subplots()
        s = plt.scatter(transformed_features[:,0], transformed_features[:,1], c=colors, cmap='jet')
        CB = fig.colorbar(s)
        CB.set_ticks(np.linspace(0,num_class-1,num_class)/num_class)
        CB.ax.set_yticklabels(class_to_use)
        plt.show()


    #Helper function for getting sample of features
    def _get_anchors(self, all_test_features, all_test_label, sample_num = 5, sample_ratio = None):
        """
        Input:
            all_test_features  
                size: N x feature_size
            all_test_label
                size: N x 1
            sample_num
                number of samples
            sample_ratio
                number of sample by ratio of corresponding class

        Output:
            anchors: 
                a tuple (features, label)
                anchors.features: 
                    visual feature
                    size: M x feature_size
                anchors.label:
                    label of each anchor 
                    size: M x 1             
        """
        #Iterate each test class
        anchors = []
        anchors_label = []

        for i in range(self.num_test_cls()):
            valid_index = np.nonzero(np.all_test_label == i)[0]
            num_index = len(valid_index)

            select_index = np.random.permutation(num_index)

            if sample_ratio is not None:
                num_select = int(num_index * sample_ratio) 
            else:
                num_select = sample_num 

                select_index = select_index[:num_index]

                anchors.append(all_test_features[select_index,:])
                anchors_label.append(all_test_label[select_index,:])

        return np.vstack(anchors), np.vstack(anchors_label)


