#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, LassoCV,Lasso,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sns.set()

1.EXPLORATION, PREPARATION ET DATAVISUALISATION DU DATASET1.1 EXPLORATION ET PREPARATION DU DATASET
# In[2]:


df = pd.read_csv('wish_data.csv').reset_index()

df.describe()


# In[3]:


df.head(5)


# In[4]:


df.info()


# In[5]:


#Préparation des colonnes pertinentes à garder
df=df.drop(["index","title","tags","shipping_is_express","merchant_name","merchant_info_subtitle","merchant_id","merchant_has_profile_picture","merchant_profile_picture","product_url","product_id","theme","crawl_month","merchant_title","product_picture"],axis=1)


# In[6]:


#Vérification des doublons
df.duplicated().sum()


# In[7]:


#Vérification des valeurs manquates
df.isnull().sum()


# In[8]:


#Remplacer les valeurs manquantes
df["product_color"] = df["product_color"].fillna("Autres")
df["has_urgency_banner"] = df["has_urgency_banner"].fillna(0)
df["urgency_text"] = df["urgency_text"].fillna("Non")
df["origin_country"] = df["origin_country"].fillna("Autres")
df["rating_five_count"] = df["rating_five_count"].fillna(0)
df["rating_four_count"] = df["rating_four_count"].fillna(0)
df["rating_three_count"] =df["rating_three_count"].fillna(0) 
df["rating_two_count"] =df["rating_two_count"].fillna(0)
df["rating_one_count"] =df["rating_one_count"].fillna(0)
df["product_variation_size_id"] =df["product_variation_size_id"].fillna(0)


# In[9]:


#Vérification des données sur notre DataFrame : est-ce propre ?
df.isnull().sum()


# In[10]:


# Exploration de la colonne 'Rating' qui est l'Évaluation des articles par  les consommateurs et arrondi.
df['rating']=df['rating'].apply(lambda x: round(x,1))
df.rating.value_counts()

1.2 DATAVISUALISATION
# In[11]:


#Répartition des articles par note
plt.hist(df.rating,bins = 5)
plt.title("Répartition des articles par note")
plt.xlabel('Note')
plt.ylabel('NB articles')
plt.show()
# L'histogramme nous montre que les articles sont pour la majeure partie notés entre 3,6 et 4,2.


# In[12]:


df.rating.describe()


# In[13]:


# Distribution des notes 'Rating'

# boîte à moustaches:  La médiane est de 3.9 
# Le min est de 1 ert le max de 5.
# Nous constatons des valeurs minimum entre 1 et 2.7.  
# La boite à moustache vient confirmer les données de l'histogramme ci desus c'est a dire que les
# les articles sont dans l'ensemble bien notés.  Il y a quelques notes inférieurs à 2.7, mais cela reste très faible.

df.boxplot(column = 'rating')
plt.title('Distribution des notes "Rating"')
plt.show()


# In[14]:


# Repartition des articles par "units sold"

fig, ax = plt.subplots(figsize=(10,3))
sns.countplot(x='units_sold',data=df)
plt.title('Répartition des articles par units_sold')
plt.show()
# La plupart des articles est catégorisée dans les unités vendues entre 100 et 20 000.


# In[15]:


#Pays d'origine des produits
sns.countplot("origin_country",data=df)
plt.title('Répartition des articles par pays d origine ')
plt.show()

# Ce graphique nous montre que la quasi totalité des produits sont originaires de Chine chez Wish.


# In[16]:


# Couleurs les plus vendues

print(df["product_color"].value_counts(normalize = True).head(10))
color = ["black", "white", "yellow", "blue", "pink", "red", "green", "grey", "purple", "Autres"]
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot("product_color", data=df[df.product_color.isin(color)],
              palette=["lightgrey", "green", "black", "yellow", "blue", "grey", "red","orange", "pink", "purple"],
              ax=ax);
plt.title('Répartition des articles par couleur d article ')
plt.show()

# la majorité des produits est vendue en 9 couleurs principalement, 
# les couleurs dominantes sont le noir et le blanc.


# In[17]:


# Distribution des prix

sns.distplot(df['price']);
plt.title("Distribution des prix")
plt.show()
# la variable 'price' a une distribution normale.
# La majorité des valeurs sont au tour de 10.  
# Ils existe des outliers à partir de 20 qui pourraient influencer la moyenne des prix


# In[18]:


# Relation entre prix et unit sold

sns.relplot(x='price',y='units_sold',kind='line',data=df)
plt.title("Relation entre prix et unit sold")
#Nous remarquons que les produits les plus vendus ont des prix<10.
#Et tous produits dont le prix>10 ont à peux près le même nombre de ventes.
# Ainsi,on pourrait se demander s'il existe une correlation linéaire entre ces deux variables?


# In[19]:


from scipy.stats import pearsonr

pd.DataFrame(pearsonr(df['units_sold'],df['price']),index=['pearson_coeff','p-value'],
             columns=['resultat_test'])
  #Le test nous renvoi une p-value<5%. Donc il accepte l'independance de ces deux variables.
  #Ainsi, l'hypthèse de non correlation linéaire évoquée par la courbe ci-dessus est confirmée..


# In[20]:


# Relation entre le prix et la note
#Affichons une courbe de relation entre les variables 'price' et 'rating'
sns.relplot(x='price', y='rating',kind='line',data=df)
plt.title("Relation entre prix et note")
#La courbe nous montre une faible correlation entre ces deux variables.


# In[21]:


pd.DataFrame(pearsonr(df['price'],df['rating']), index=['pearson_ceff','p-value'],
             columns=['resultat_test'])
#Le test nous confirme qu'il y a une faible corrélation entre les deux variables.


# In[22]:


# Distribution des notes 'Rating'

# boîte à moustaches:  La médiane est de 3.9 
# Le min est de 1 ert le max de 5.
# Nous constatons des valeurs minimum entre 1 et 2.7.  
# La boite à moustache vient confirmer les données de l'histogramme ci desus c'est a dire que les
# les articles sont dans l'ensemble bien notés.  Il y a quelques notes inférieurs à 2.7, mais cela reste très faible.

df.boxplot(column = 'rating')
plt.title('Distribution des notes "Rating"')
plt.show()


# In[23]:


# Repartition des articles par "units sold"

fig, ax = plt.subplots(figsize=(10,3))
sns.countplot(x='units_sold',data=df)
plt.title('Répartition des articles par units_sold')
plt.show()
# La plupart des articles est catégorisée dans les unités vendues entre 100 et 20 000.


# In[24]:


#Pays d'origine des produits
sns.countplot("origin_country",data=df)
plt.title('Répartition des articles par pays d origine ')
plt.show()

# Ce graphique nous montre que la quasi totalité des produits sont originaires de Chine chez Wish.


# In[25]:


# Couleurs les plus vendues

print(df["product_color"].value_counts(normalize = True).head(10))
color = ["black", "white", "yellow", "blue", "pink", "red", "green", "grey", "purple", "Autres"]
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot("product_color", data=df[df.product_color.isin(color)],
              palette=["lightgrey", "green", "black", "yellow", "blue", "grey", "red","orange", "pink", "purple"],
              ax=ax);
plt.title('Répartition des articles par couleur d article ')
plt.show()

# la majorité des produits est vendue en 9 couleurs principalement, 
# les couleurs dominantes sont le noir et le blanc.


# In[26]:


# Distribution des prix

sns.distplot(df['price']);
plt.title("Distribution des prix")
plt.show()
# la variable 'price' a une distribution normale.
# La majorité des valeurs sont au tour de 10.  
# Ils existe des outliers à partir de 20 qui pourraient influencer la moyenne des prix


# In[27]:


# Relation entre prix et unit sold

sns.relplot(x='price',y='units_sold',kind='line',data=df)
plt.title("Relation entre prix et unit sold")
#Nous remarquons que les produits les plus vendus ont des prix<10.
#Et tous produits dont le prix>10 ont à peux près le même nombre de ventes.
# Ainsi,on pourrait se demander s'il existe une correlation linéaire entre ces deux variables?


# In[28]:


from scipy.stats import pearsonr

pd.DataFrame(pearsonr(df['units_sold'],df['price']),index=['pearson_coeff','p-value'],
             columns=['resultat_test'])
  #Le test nous renvoi une p-value<5%. Donc il accepte l'independance de ces deux variables.
  #Ainsi, l'hypthèse de non correlation linéaire évoquée par la courbe ci-dessus est confirmée..


# In[29]:


# Relation entre le prix et la note
#Affichons une courbe de relation entre les variables 'price' et 'rating'
sns.relplot(x='price', y='rating',kind='line',data=df)
plt.title("Relation entre prix et note")
#La courbe nous montre une faible correlation entre ces deux variables.


# In[30]:


pd.DataFrame(pearsonr(df['price'],df['rating']), index=['pearsonr_coeff','p-value'],
             columns=['resultat_test'])
#Le test nous confirme qu'il y a une faible corrélation entre les deux variables.


# In[31]:


# Repartition des articles par unit sold selon l'existence d'une publicité
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot("units_sold", data=df,hue=df.uses_ad_boosts)
plt.title("Repartition des articles par unit sold selon l'existence d'une publicité")

#L'existance d'une publicité sur un produit ne semble pas avoir d'impact sur la quantité vendue
#En effet nous voyons à travers ce graphique que le nombre d'articles répartis dans les units solds est bien
#bien supérieur dans la catégorie sans pub vs avec pub


# In[89]:


# Création de la variable 'ecart' entre le prix et le detail du prix
df['Ecart_price'] = df.price-df.retail_price
#Affichons l'influence de cette variable sur le nombre d'unité vendues
sns.relplot(x='Ecart_price', y='units_sold',kind='line',data=df)
plt.title("Relation entre prix et units sold")
#La courbe nous montre une faible correlation entre ces deux variables.
# Ceci est un resultat qu'on attendais puisque le test de pearson  fait plus haut nous confirmait que la variable 'price' n'etait
# pas correlée avec la variable 'units_sold'.


# In[33]:


# Finissons cette exploration par l'étude de la correlation entre toutes les variables de df
corr = df.corr()
fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')

Conclusion de la partie 1 Exploration, préparation et datavisualisation du dataset
Cette première étape nous montre peu de corrélation entre les variables.
Nous remarquons une forte correlation entre la variable 'units_sold' et les variables 'rating_count','rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count', 'rating_one_count'.
C'est ce que nous attendons, parcque les articles sont notés après leur achat
Nous remarquons aussi que la variable 'price' est fortement correlée avec la variable 'shipping_option_price', ce qui est tout à fait normale
Ce pendant il n'y a pas de correlation entre la variable 'units_sold' et 'rating' ce qui suppose que la vente d'un article, ne depend pas de sa côte?
# 1.MACHINE LEARNING
# 1.1 REGRESSION

# In[34]:


#Certaines colonnes ne vont pas être utiles pour faire des modèles de régression.
df=df.drop(["shipping_option_name","urgency_text","title_orig","currency_buyer","origin_country","product_color","product_variation_size_id","Ecart_price"],axis=1)
df.info()
#Nous allons travailler avec 23 variables 


# In[35]:


##Normalisons les données
scaler = preprocessing.StandardScaler().fit(df)

df[df.columns] = pd.DataFrame(scaler.transform(df), index= df.index)
df


# In[36]:


# Nous remarquons depuis la matrice de corrélation faite précédemment plusieurs éléments:
#1) units_sold est très corrélé avec le rating_count (le nombre de notes)
#2) le prix est corrélé avec shipping_option_price(le prix des options de livraison)


# In[37]:


# Séparons les variables target et features
target = df['units_sold']
data = df.drop('units_sold',axis=1)

#Partagons aleatoirement nos données en ensemble d'entrainement et de test
X_train,X_test,y_train,y_test = train_test_split(data, target, test_size = 0.2, random_state= 789)


# **REGRESSION LINEAIRE**

# In[38]:



# Intensions un modèle de regression linéaire et ajustons-le aux données d'entrainement
lr = LinearRegression()
lr.fit(X_train, y_train)


# In[39]:


#Affichons l'intercept, et les coefficients estimées de chaque variable
coefs = list(lr.coef_)
coefs.insert(0, lr.intercept_)

feats = list(data.columns)
feats.insert(0,'intercept')

pd.DataFrame({'valeur estimée':coefs}, index=feats)


# In[40]:


# Affichons le score du modele et le coefficient obtenue par validation croisée
print("Le coefficient de determination:",lr.score(X_train,y_train))
print("Le score du model sur l'ensemble test:", lr.score(X_test,y_test))

#Le score obtenue sur l'echantillon d'apprentissage est correct, mais le score obtenue sur le test montre qu'il y a peut être un surapprentissage du modele.
#Essayons d'ameliorer le model en choisissant les variables explicatives les plus significatifs possible .


# In[41]:


#prédiction du modèle

pred_lrtest = lr.predict(X_test)
plt.scatter(pred_lrtest, y_test)
plt.plot((y_test.min(),y_test.max()), (y_test.min(),y_test.max()))
plt.show()

#la droite semble ajustée au début sauf pour certaines valeur comme nous pouvons observer


# In[42]:


# Selection des variables les plus correlées a 'units_sold'
s_feats = ['rating_count','rating_one_count','rating_two_count','rating_three_count','rating_four_count','rating_five_count']

#Créeons un nouveau modele et ajustons le avec les variable s_feats de X_train
lr2 = LinearRegression()
lr2.fit(X_train[s_feats], y_train)


# In[43]:


# Affichons le score du modele

print("Le score du modele sur l'echantillon d'entrainement:", lr2.score(X_train[s_feats], y_train))
print("Le score du modele sur l'echantillon test:", lr2.score(X_test[s_feats],y_test))

#Le modèle présente presque les mêmes résultats que le précédent.


# In[44]:


#affichons a present les resultats de notre prediction


moy = scaler.mean_[-1]
ec = scaler.scale_[-1]
print("moyenne :", moy)
print("ecart-type", ec)

pd.DataFrame({'ventes_observées': (y_test*ec)+moy, 'ventes_predites' : (pred_lrtest*ec)+moy}, index = X_test.index).head(7)

# Le modèle de prédiction semble proche de la réalité


# LASSO

# In[45]:


# Intensions un modèle de Lasso et ajustons-le aux données d'entrainement
lasso_r = Lasso(alpha=1)

lasso_r.fit(X_train, y_train)

print("Lasso_coef",lasso_r.coef_)
##Tous les coefficients sont nuls. Manifestement, la valeur α=1.0α=1.0 ne convient pas à nos données.


# In[46]:


from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)

lasso_reg.fit(X_train, y_train)
print("Lasso_coef",lasso_reg.coef_)


# In[47]:


#graphique représentant la valeur estimée du coefficient pour chaque variable"

lasso_coef = lasso_reg.coef_

plt.figure(figsize=(12,3))
plt.plot(range(len(data.columns)), lasso_coef)
plt.xticks(range(len(data.columns)), data.columns.values, rotation=70)
plt.show()
##Seulement deux variables ont été retenus


# In[48]:


#Affichons le score du modele

print("Le score du modele sur l'echantillon d'entrainement:", lasso_reg.score(X_train, y_train))
print("Le score du modele sur l'echantillon test:", lasso_reg.score(X_test,y_test))

#Surapprentissage apparent


# In[49]:


#Prédiction aux valeurs ajustées
lasso_pred_train = lasso_reg.predict(X_train)
lasso_pred_test = lasso_reg.predict(X_test)

print("mse train:", mean_squared_error(lasso_pred_train, y_train))
print("mse test:", mean_squared_error(lasso_pred_test, y_test))

##Les résultats ne semblent pas mieux que la Régression Linéaire


# In[50]:


#Utilisons le LassoPath 

from sklearn.linear_model import lasso_path

mes_alphas = (0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0)

alpha_path, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=mes_alphas)

coefs_lasso.shape


# In[51]:


import matplotlib.cm as cm

plt.figure(figsize=(10,7))

for i in range(coefs_lasso.shape[0]):
    plt.plot(alpha_path, coefs_lasso[i,:], '--')

plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso path')
plt.show()


# In[52]:


from sklearn.linear_model import LassoCV

model_lasso = LassoCV(cv=10).fit(X_train, y_train)

alphas = model_lasso.alphas_

plt.figure(figsize = (10,8))

plt.plot(alphas, model_lasso.mse_path_, ':')

plt.plot(alphas, model_lasso.mse_path_.mean(axis=1), 'k',
         label='Moyenne', linewidth=2)

plt.axvline(model_lasso.alpha_, linestyle='--', color='k',
            label='alpha: estimation CV')

plt.legend()

plt.xlabel('Alpha')
plt.ylabel('Mean square error')
plt.title('Mean square error pour chaque échantillon ')
plt.show()


# In[53]:



pred_lassotest = model_lasso.predict(X_test)

print("score test:", model_lasso.score(X_test, y_test))
print("mse test:", mean_squared_error(pred_lassotest, y_test))

##Les résultats sont mieux sur l'échantillon de test


# In[54]:


moy = scaler.mean_[-1]
ec = scaler.scale_[-1]
print("moyenne :", moy)
print("ecart-type", ec)

pd.DataFrame({'ventes_observées': (y_test*ec)+moy, 'ventes_predites' : (pred_lassotest*ec)+moy}, index = X_test.index).head(7)


# **ELASTIC NET**

# In[55]:


# Intensions un dernier modèle d'ElasticNet et ajustons-le aux données d'entrainement
model_en = ElasticNetCV(cv=8, l1_ratio = (0.1, 0.25, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.99), 
                        alphas= (0.001,0.01,0.02,0.025,0.05,0.1,0.25,0.5,0.8,1.0))
model_en.fit(X_train, y_train)


# In[56]:


coeffs = list(model_en.coef_)
coeffs.insert(0, model_en.intercept_)
feats = list(data.columns)
feats.insert(0, 'intercept')

pd.DataFrame({'valeur estimée': coeffs}, index = feats)


# In[57]:


#Affichons le score du modele

print("Le score du modele sur l'echantillon d'entrainement:", model_en.score(X_train, y_train))
print("Le score du modele sur l'echantillon test:", model_en.score(X_test,y_test))


# In[58]:


#Prédiction aux valeurs ajustées
pred_entrain = model_en.predict(X_train)
pred_entest = model_en.predict(X_test)

print("rmse train:", np.sqrt(mean_squared_error(y_train, pred_entrain)))
print('rmse test:', np.sqrt(mean_squared_error(y_test, pred_entest)))

##Les résultats ne semblent pas mieux que la Régression Linéaire


# In[59]:


moy = scaler.mean_[-1]
ec = scaler.scale_[-1]
print("moyenne :", moy)
print("ecart-type", ec)

pd.DataFrame({'ventes_obsérvés': (y_test*ec)+moy, 'ventes_predits' : (pred_entest*ec)+moy}, index = X_test.index).head(7)


#  <h2>MACHINE LEARNING</h2>  
#  <h3>CLASSIFICATION</h3>
# <h4>Algorithme de machine learning</h4> 
# Nous avons constatés que la variable 'Units sold' correspond a une classification de quantités d'unitées vendues. C'est pourquoi nous utiliserons les méthode de classifications de machine learning et non pas de regression.
# 
# Nous construirons dans un premeir temps un Arbres de décision pour connaitre l'importance des variables.
# 
# Puis nous allons comparer plusieurs méthodes d'apprentissages
# K-plus proches voisins, SVM, RandomForest. 
# Puis nous crérons une méthode d'ensemble

# In[60]:


from sklearn import model_selection
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[61]:


#inverser la normalisation des données --> ramener les données à l'origine


# In[62]:


df2 = pd.read_csv('data_wish.csv').reset_index()
#Préparation des colonnes pertinentes à garder
df2=df2.drop(["index","title","tags","shipping_is_express","merchant_name","merchant_info_subtitle","merchant_id","merchant_has_profile_picture","merchant_profile_picture","product_url","product_id","theme","crawl_month","merchant_title","product_picture"],axis=1)
#Remplacer les valeurs manquantes
df2["product_color"] = df2["product_color"].fillna("Autres")
df2["has_urgency_banner"] = df2["has_urgency_banner"].fillna(0)
df2["urgency_text"] = df2["urgency_text"].fillna("Non")
df2["origin_country"] = df2["origin_country"].fillna("Autres")
df2["rating_five_count"] = df2["rating_five_count"].fillna(0)
df2["rating_four_count"] = df2["rating_four_count"].fillna(0)
df2["rating_three_count"] =df2["rating_three_count"].fillna(0) 
df2["rating_two_count"] =df2["rating_two_count"].fillna(0)
df2["rating_one_count"] =df2["rating_one_count"].fillna(0)
df2["product_variation_size_id"] =df2["product_variation_size_id"].fillna(0)
# Exploration de la colonne 'Rating' qui est l'Évaluation des articles par  les consommateurs et arrondi.
df2['rating']=df2['rating'].apply(lambda x: round(x,1))
df2=df2.drop(["shipping_option_name","urgency_text","title_orig","currency_buyer","origin_country","product_color","product_variation_size_id"],axis=1)


# In[63]:


# Séparons les variables target et features
target2 = df2['units_sold']
data2 = df2.drop('units_sold',axis=1)

#Partagons aleatoirement nos données en ensemble d'entrainement et de test
X_train, X_test, y_train, y_test = train_test_split(data2, target2, test_size = 0.2)

#Centrer et réduire les variables explicatives des deux échantillons de manière adéquate.
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#  **ARBRE DE DECISION**

# In[72]:


dt =  DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=321)
dt.fit(X_train_scaled, y_train)


# In[78]:


y_pred = dt.predict(X_test_scaled)
print(dt.score(X_test,y_test))
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])
# Les prédictions faites sur l'echantillons test avec un taux de 73% de bonnes predictions sont encouragentes.


# In[76]:


# Affichons les 6 variables les plus importantes, ainsi que leurs importances respectives

feats ={}
for feature, importance in zip(data2.columns, dt.feature_importances_):
    feats[feature]= importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0:'Importance'})
importances.sort_values(by='Importance',ascending=False).head(6)


# *Ces variables sont effectivement les variables les plus correlées à notre variable cible('units_sold'). 
# Ce qui confirme l'hypothèse de corrélation de la matrice de correlation de la partie visualisation*
# 

# *   Notre problematique consiste à créer un model de prédiction du nombre d'unités vendues pour chaque produit referencier. 
# * Par conséquant,il n'est pas absolument nécessaire d'avoir une prediction exacte. Donc ce modele est tout a fait adequat par le faite qu'il soit facilement **interpretable** en affichant l'arbre de décision. 

# **KNN**

# In[64]:


# Création du classificateur et construction du modèle sur les données d'entraînement
knn = neighbors.KNeighborsClassifier(n_neighbors=8, metric='minkowski')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# **SVM**

# In[65]:



#Entrainement du modèle
clf = svm.SVC()

parametres = {'C':[0.1,1,10],'kernel':['rbf','linear', 'poly'], 'gamma':[0.001, 0.1, 0.5]}
grid_clf = model_selection.GridSearchCV(estimator =clf,param_grid=parametres)

grille = grid_clf.fit(X_train_scaled,y_train)

print(pd.DataFrame.from_dict(grille.cv_results_).loc[:,["params","mean_test_score"]]) 


# In[66]:


#Meilleur score : {'C': 10, 'gamma': 0.001, 'kernel': 'linear'} 


# In[67]:


y_pred = grid_clf.predict(X_test_scaled)


# In[68]:


pd.crosstab(y_test, y_pred, rownames=["Classe réelle"], colnames = ["Classe prédite"])


# **REGRESSION LOGISTIQUE**

# In[69]:


lgr = linear_model.LogisticRegression(C=1.0)
lgr.fit(X_train_scaled,y_train)


# In[70]:


y_pred = lgr.predict(X_test_scaled)
pd.crosstab(y_test, y_pred, rownames=['classe reelle'], colnames=['classe predite'])


# **RANDOM FOREST**

# In[71]:


rf = ensemble.RandomForestClassifier(n_jobs = -1, random_state = 321)
rf.fit(X_train_scaled, y_train) 
y_pred = rf.predict(X_test_scaled)
a= np.arange(2,31,2)

parametres= [{'n_estimators': [10, 50, 100, 150, 200, 250],
                   'min_samples_leaf': ['a'], 
                   'max_features': ['sqrt', 'log2']}]


grid_rf = model_selection.GridSearchCV(estimator=rf.fit, param_grid=parametres)
grille = rf.fit(X_train_scaled,y_train)

pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])


# ***Le modèle de Random Forest semble être le meilleur classificateur parmi les modèles testés!***
# 
