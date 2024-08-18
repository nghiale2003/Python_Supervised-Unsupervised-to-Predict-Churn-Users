# Python_Supervised-Unsupervised-to-Predict-Churn-Users
Using ML( Supervised &amp; Unsupervised Learning) to predict churned users; segment these churned users into groups

# Question:
One ecommerce company has a project on predicting churned users in order to offer potential promotions. 
An attached file is the dataset that is offered by the company (churn_predict.csv). You will using these dataset to answer below questions: 
1. What are the patterns/behavior of churned users? What are your suggestions to the company to reduce churned users. 
2. Build the Machine Learning model for predicting churned users. (fine tuning) 
3. Based on the behaviors of churned users, the company would like to offer some special promotions for them. 
Please segment these churned users into groups. What are the differences between groups?

# Processing: It is just summary the workflow, if you want to see more details can click the source i attached
## I. EDA:
### 1. Check info & missing value
    raw_data.info()

![image](https://github.com/user-attachments/assets/e4b3a5a8-6946-4db9-b408-884b32e490a9)

=> Need to fill the missing values: Tenure, WarehouseToHome , HourSpendOnApp, OrderAmountHikeFromlastYear , CouponUsed, OrderCOunt, DaySinceLast Order

    #Fill missing value by median in that columns
    
    need_fill_median= ['Tenure', 'WarehouseToHome' , 'HourSpendOnApp', 'OrderAmountHikeFromlastYear' , 'CouponUsed', 'OrderCount', 'DaySinceLastOrder']
    
    raw_data[need_fill_median]=raw_data[need_fill_median].fillna(raw_data[need_fill_median].median())
    
    raw_data.info()

![image](https://github.com/user-attachments/assets/84df442b-473c-48e9-b6ef-6de5c6e96a1e)

### 2. Check duplicate

### 3. Check imbalanced value

    label_ratio= raw_data['Churn'].value_counts(normalize=True)

![image](https://github.com/user-attachments/assets/ef47a4e7-a8f8-4d3c-87bd-51f67df3d501)

=> Still apply model and adjust if the accuracy score is too low

### 4.Encoding and Transforming

#### Encoding
**#Drop columns: CustomerId because it's just a primary key to identify**

    data_transform= raw_data.copy().drop('CustomerID', axis=1)
    
    cate_columns= data_transform.select_dtypes(include=['category']).columns.tolist()
    
    encode_df= pd.get_dummies(data_transform,columns=cate_columns, drop_first=True, dtype='int')

![image](https://github.com/user-attachments/assets/8e04befd-eaea-4116-877f-edfdbb9e5530)

#### Transforming

    from sklearn.preprocessing import MinMaxScaler
    
    **#Scaler features:**
    
    scaler=MinMaxScaler()
    
    model=scaler.fit(encode_df)
    
    scaled_data=model.transform(encode_df)
    
    scaled_df=pd.DataFrame(encode_df, columns=encode_df.columns.tolist())

![image](https://github.com/user-attachments/assets/92223ad4-12cc-46bd-96a9-372b48124fa3)

### 5. Apply RandomForest to identify related features

    from sklearn.model_selection import train_test_split
    
    X= scaled_df.drop('Churn', axis=1)
    
    y= scaled_df['Churn']

**#Split the dataset**

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.33, random_state=42)

**#Training model**

    from sklearn.ensemble import RandomForestClassifier
    
    model= RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)

**#Evalue the model**

    accuracy_before= model.score(X_test, y_test)
    
    print(f'Accuracy before feature selection: {accuracy_before:.2f}')

![image](https://github.com/user-attachments/assets/5b135a6a-0517-4db8-8e2c-c2cd8226bc45)

**#Extract feature importances**

    importances= model.feature_importances_
    
    feature_names= X.columns.tolist()
    
    feature_importances= pd.DataFrame({'Feature': feature_names, 'Importance': importances})

**#Ranking features by importance**

    feature_importances= feature_importances.sort_values(by='Importance', ascending=False)

![image](https://github.com/user-attachments/assets/6e89dd83-cb56-4d39-9a8e-320fb7aa086b)

# Select  importance features > 0.01

    top_features= feature_importances[feature_importances['Importance'] > 0.01]['Feature'].values
    
    selected_scaled_df=scaled_df[top_features.tolist()+['Churn']]

## II. Build Supervised Learning to predict Churn Users

### 1. Check accuracy the previous model

    y_pred= model_selected.predict(X_test)
    
    print(classification_report(y_pred, y_test))

![image](https://github.com/user-attachments/assets/a515177f-eba0-4e38-ad26-42c1a8791e4d)

**=>I choose the RandomForest and Imporve the accuracy of the model**

### 2. Fine tuning

    param_grid = {
        'n_estimators': [75, 100, 125,150],
        *max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9, None],
        'max_leaf_nodes': [3, 6, 9, None],
    }
    
    grid_search = GridSearchCV(RandomForestClassifier(),
                               param_grid=param_grid)
                               
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_estimator_)

**=> Final model RandomForestClassifier(max_features=None, n_estimators=75)**

    model_grid = RandomForestClassifier( max_features=None
                                          , n_estimators=75)
    model_grid.fit(X_train, y_train)
    
    y_pred_grid = model_grid.predict(X_test)
    
    print(classification_report(y_pred_grid, y_test))

## III. Build Unsupervised to classify churn users's segemnt

### 1. Select data only churned user

    raw_data_churn= raw_data[raw_data['Churn']==1]
    
    raw_data_churn.head()
    
    scaled_df_churn= scaled_df[scaled_df['Churn']==1].drop('Churn', axis=1)
    
    scaled_df_churn.head()

### 2. Dimension Reduction

    from sklearn.decomposition import PCA
    
    pca=PCA(n_components=10)
    
    pca.fit(scaled_df_churn)
    
    PCA_df=pd.DataFrame(pca.transform(scaled_df_churn), columns=['PC1', 'PC2', 'PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
    
    PCA_df.head()

![image](https://github.com/user-attachments/assets/2dca5518-e3ea-4b79-9bde-d373b3423133)

    pca.explained_variance_ratio_

![image](https://github.com/user-attachments/assets/86ea2db3-6427-400a-8144-28d1faad67fb)

**=> variance ratio is enough low to apply the model**

### 3. Apply K-Mean Model

#### Choose K

    from sklearn.cluster import KMeans
    
    import matplotlib.pyplot as plt, numpy as np
    
    from mpl_toolkits.mplot3d import Axes3D
    
    from sklearn.cluster import AgglomerativeClustering
    
    from matplotlib.colors import ListedColormap
    
    ss = []
    
    max_clusters = 10
    
    for i in range(1, max_clusters+1):
    
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        
        kmeans.fit(PCA_df)
        
        #Inertia method returns WCSS for that model
        
        ss.append(kmeans.inertia_)

**Plot the Elbow method**

      plt.figure(figsize=(10,5))
      plt.plot(range(1, max_clusters+1), ss, marker='o', linestyle='--')
      plt.title('Elbow Method')
      plt.xlabel('Number of clusters')
      plt.ylabel('WCSS')
      plt.show()

![image](https://github.com/user-attachments/assets/44290b72-34c7-488b-924e-75c42347345c)

**=> Choose K=4**

#### Apply K-Mean Model with K=4

      kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
      predicted_labels = kmeans.fit_predict(PCA_df)
      
      PCA_df['clusters']=predicted_labels
      PCA_df.head()

      raw_data_churn['clusters']=predicted_labels
      raw_data_churn.head()

      #Plotting the clusters
      fig = plt.figure(figsize=(10,8))
      ax = plt.subplot(111, projection='3d', label="bla")
      ax.scatter(PCA_df['PC1'], PCA_df['PC2'], PCA_df['PC3'], s=40, c=PCA_df["clusters"], marker='o')
      ax.set_title("The Plot Of The Clusters")
      plt.show()

![image](https://github.com/user-attachments/assets/ae7800c3-ae93-4f85-b181-fbeaacbbf909)

#### Features of each group

**Group 0:** This group seems to have a low frequency of purchases, but when they do purchase, the increase is significant. They live mostly in higher-tier cities and spend less time on the app.

**Group 1:** This group has medium purchase frequency with high order increase. They are distributed across all city tiers and spend a moderate amount of time on the app.

**Group 2:** This group is characterized by low purchase frequency with significant order increase, high satisfaction scores, and considerable time spent on the app.

**Group 3:** This group has the highest purchase frequency with the least increase in orders. They mostly live in higher and mixed-tier cities and spend the least time on the app.















