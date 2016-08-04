#https://github.com/sujitpal/nltk-examples/blob/master/src/semantic/short_sentence_similarity.py
print('***** import packages')
import pandas as pd
print('******read corpus *******************')
data = pd.read_csv("/home/1060929/Suresh/Personal/RnD/ParaPhrasing/MSRParaphraseCorpus/msr_paraphrase_train.csv",sep = '\\t')

#print(data.shape)
#print(type(data))
# ********* basic statistics *********
#print(data.groupby(['Quality']).count())
#unbalanced data 0 - 32% 1- 68%

# ******** TDM ***********************
print('**** prepare data for TDM *****')
for i in range(1,10):
	txt1 = data.ix[i,3]
	txt2 = data.ix[i,4]
	tmpdf = pd.DataFrame({'txt':[txt1,txt2]})
	
	#print(tmpdf)
	import nltk
	from sklearn.feature_extraction.text import CountVectorizer
	countvec = CountVectorizer()
	#print(tmpdf.txt)
	#print(pd.DataFrame(countvec.fit_transform(tmpdf.txt).toarray(), columns=countvec.get_feature_names()))
	#print(type(TDM))
	#print(TDM)
	
	print('********* similarity calculations ******')
	#cosine similarity
	TDM = pd.DataFrame(countvec.fit_transform(tmpdf.txt).toarray(), columns=countvec.get_feature_names())
	from scipy.spatial.distance import cosine
	print('cosine similarity with TDM')
	print(1 - cosine(TDM.ix[0,0:], TDM.ix[1,0:]))
	print('cosine similarity with tfidf')
	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf_nostop = TfidfVectorizer().fit_transform(tmpdf.txt)
	tfidf_withstop = TfidfVectorizer(ngram_range=(1, 1), stop_words='english').fit_transform(tmpdf.txt)
	#print(tfidf_nostop.shape)
	#print(tfidf_withstop.shape)
	#print(type(tfidf_nostop))
	from sklearn.metrics.pairwise import linear_kernel
	print(linear_kernel(tfidf_nostop[0], tfidf_nostop[1]).flatten())
	print(linear_kernel(tfidf_withstop[0], tfidf_withstop[1]).flatten())
	print('cosine similarity with LSA SVD appraoch')
	from sklearn.decomposition import TruncatedSVD
	svd1 = TruncatedSVD(n_components = int(tfidf_nostop.shape[1])-1)
	svdMatrix_nostop = svd1.fit_transform(tfidf_nostop)
	svd2 = TruncatedSVD(n_components = int(tfidf_withstop.shape[1])-1)	
	svdMatrix_withstop = svd2.fit_transform(tfidf_withstop)
	print(1 - cosine(svdMatrix_nostop[0], svdMatrix_nostop[1]))	
	print(1 - cosine(svdMatrix_withstop[0], svdMatrix_withstop[1]))
	
print('*************NEXT iteration******************')
#add wordnet & tfidf
print('******running pos tagging *********')
txt01 = data.ix[1,3]
txt02 = data.ix[1,4]
print(txt01)
print(txt02)
#pd.DataFrame({'txt':[txt1,txt2]})
sent1  =  nltk.sent_tokenize(txt01)
sent2  =  nltk.sent_tokenize(txt02)
loftags = []
for s in sent1:
	d = nltk.word_tokenize(s)   
	print(nltk.pos_tag(d))
for s in sent2:
	d = nltk.word_tokenize(s)   
	print(nltk.pos_tag(d))
