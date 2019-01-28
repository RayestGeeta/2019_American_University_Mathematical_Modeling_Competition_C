from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor

wbk = xlsxwriter.Workbook('All_feature.xlsx')
sheet = wbk.add_worksheet('Data')

Al = ['GradientBoosting', 'RandomForest', 'Bagging', 'Decision', 'KNN']

sheet.write(0, 0, 'mean_squared_error')
sheet.write(8, 0, 'explained_variance_score')

for i in range(len(Al)):
    sheet.write(0, i+1, Al[i])
    
for i in range(2010,2017):
    sheet.write(i%2009, 0, i)
    sheet.write(i%2001, 0, i)
for k in range(0, 7):
    
    String = '2eco1' + str(k) + '.csv'
    Df = pd.read_csv(String)
    Datas = Df.values
    label = []
    sorts = []
    names = [i for i in range(0, len(Df.columns[1:-1]))]
    for i in Datas:
        label.append(i[-1])
        sorts.append(i[0])
    datas = np.delete(Datas, 0, axis = 1)
    datas = np.delete(datas, -1, axis = 1)
    
    X = datas
    
    mean_squared = {}
    explained_variance = {}
    for i in range(10):
        for j in range(1,5):
            X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=1-0.1*j)
            
            
            gbt = GradientBoostingRegressor(n_estimators=100)
            gbt.fit(X_train, y_train)
            
            if 'GradientBoosting' not in mean_squared:
                mean_squared['GradientBoosting'] = round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] = round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
            else:
                mean_squared['GradientBoosting'] += round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] += round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
                
            
            
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            
            if 'RandomForest' not in mean_squared:
                mean_squared['RandomForest'] = round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] = round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
            else:
                mean_squared['RandomForest'] += round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] += round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
                
                
                
            bagging_clf = BaggingRegressor(DecisionTreeRegressor(), 
                                n_estimators = 100, max_samples = 30,
                                bootstrap = True, oob_score = True)
            bagging_clf.fit(X_train,y_train)
            
             
            if 'Bagging' not in mean_squared:
                mean_squared['Bagging'] = round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] = round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)
            else:
                mean_squared['Bagging'] += round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] += round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)

           

                
                
            d_clf = DecisionTreeRegressor()
            d_clf.fit(X_train, y_train)
            
            if 'Decision' not in mean_squared:
                mean_squared['Decision'] = round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] = round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)
            else:
                mean_squared['Decision'] += round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] += round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)

                
                
                
            k_clf = KNeighborsRegressor(4)
            k_clf.fit(X_train, y_train)
            
            if 'KNN' not in mean_squared:
                mean_squared['KNN'] = round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] = round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)
            else:
                mean_squared['KNN'] += round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] += round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)

                
    print(k)
    for i in mean_squared:
        mean_squared[i] = round(mean_squared[i]/40,2)
        explained_variance[i] = round(explained_variance[i]/40,4)
    
    for i in range(len(Al)):
        sheet.write(k+1, i+1, mean_squared[Al[i]])
        sheet.write(k+9, i+1, explained_variance[Al[i]])
        
    
    print('mean_squared_error:', mean_squared)
    print()
    print('explained_variance_score', explained_variance)
    print()
    
wbk.close()

import os
path = '/Users/zhenshihao/Desktop/2018_MCMProblemC_DATA/Useful-Data'
os.chdir(path)

wbk = xlsxwriter.Workbook('Special_feature.xlsx')
sheet = wbk.add_worksheet('Data')

Al = ['GradientBoosting', 'RandomForest', 'Bagging', 'Decision', 'KNN']

sheet.write(0, 0, 'mean_squared_error')
sheet.write(8, 0, 'explained_variance_score')

for i in range(len(Al)):
    sheet.write(0, i+1, Al[i])
    
for i in range(2010,2017):
    sheet.write(i%2009, 0, i)
    sheet.write(i%2001, 0, i)
for k in range(0, 7):
    
    String = '2eco1' + str(k) + '.csv'
    Df = pd.read_csv(String)
    Datas = Df.values
    label = []
    sorts = []
    names = [i for i in range(0, len(Df.columns[1:-1]))]
    for i in Datas:
        label.append(i[-1])
        sorts.append(i[0])
    datas = np.delete(Datas, 0, axis = 1)
    datas = np.delete(datas, -1, axis = 1)
    
    print(len(datas))
    X = datas[:,sort[k]]
    
    
    mean_squared = {}
    explained_variance = {}
    for i in range(10):
        for j in range(1,5):
            X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=1-0.1*j)
            
            
            gbt = GradientBoostingRegressor(n_estimators=100)
            gbt.fit(X_train, y_train)
            
            if 'GradientBoosting' not in mean_squared:
                mean_squared['GradientBoosting'] = round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] = round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
            else:
                mean_squared['GradientBoosting'] += round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] += round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
                
            
            
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            
            if 'RandomForest' not in mean_squared:
                mean_squared['RandomForest'] = round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] = round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
            else:
                mean_squared['RandomForest'] += round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] += round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
                
                
                
            bagging_clf = BaggingRegressor(DecisionTreeRegressor(), 
                                n_estimators = 100, max_samples = 30,
                                bootstrap = True, oob_score = True)
            bagging_clf.fit(X_train,y_train)
            
             
            if 'Bagging' not in mean_squared:
                mean_squared['Bagging'] = round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] = round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)
            else:
                mean_squared['Bagging'] += round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] += round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)

           

                
                
            d_clf = DecisionTreeRegressor()
            d_clf.fit(X_train, y_train)
            
            if 'Decision' not in mean_squared:
                mean_squared['Decision'] = round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] = round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)
            else:
                mean_squared['Decision'] += round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] += round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)

                
                
                
            k_clf = KNeighborsRegressor(4)
            k_clf.fit(X_train, y_train)
            
            if 'KNN' not in mean_squared:
                mean_squared['KNN'] = round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] = round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)
            else:
                mean_squared['KNN'] += round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] += round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)

                
    print(k)
    for i in mean_squared:
        mean_squared[i] = round(mean_squared[i]/40,2)
        explained_variance[i] = round(explained_variance[i]/40,4)
    
    for i in range(len(Al)):
        sheet.write(k+1, i+1, mean_squared[Al[i]])
        sheet.write(k+9, i+1, explained_variance[Al[i]])
        
    
    print('mean_squared_error:', mean_squared)
    print()
    print('explained_variance_score', explained_variance)
    print()
    
wbk.close()

import os
path = '/Users/zhenshihao/Desktop/2018_MCMProblemC_DATA/Useful-Data'
os.chdir(path)

wbk = xlsxwriter.Workbook('Some_feature.xlsx')
sheet = wbk.add_worksheet('Data')

Al = ['GradientBoosting', 'RandomForest', 'Bagging', 'Decision', 'KNN']

sheet.write(0, 0, 'mean_squared_error')
sheet.write(8, 0, 'explained_variance_score')

for i in range(len(Al)):
    sheet.write(0, i+1, Al[i])
    
for i in range(2010,2017):
    sheet.write(i%2009, 0, i)
    sheet.write(i%2001, 0, i)
for k in range(0, 7):
    
    String = '2eco1' + str(k) + '.csv'
    Df = pd.read_csv(String)
    Datas = Df.values
    label = []
    sorts = []
    names = [i for i in range(0, len(Df.columns[1:-1]))]
    for i in Datas:
        label.append(i[-1])
        sorts.append(i[0])
    datas = np.delete(Datas, 0, axis = 1)
    datas = np.delete(datas, -1, axis = 1)
    
    print(len(datas))
    X = datas[:,[3,5,9,16]]
    
    
    mean_squared = {}
    explained_variance = {}
    for i in range(10):
        for j in range(1,5):
            X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=1-0.1*j)
            
            
            gbt = GradientBoostingRegressor(n_estimators=100)
            gbt.fit(X_train, y_train)
            
            if 'GradientBoosting' not in mean_squared:
                mean_squared['GradientBoosting'] = round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] = round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
            else:
                mean_squared['GradientBoosting'] += round(sm.mean_squared_error(y_test,gbt.predict(X_test)),2)
                explained_variance['GradientBoosting'] += round(sm.explained_variance_score(y_test,gbt.predict(X_test)),4)
                
            
            
            rf = RandomForestRegressor(n_estimators=100)
            rf.fit(X_train, y_train)
            
            if 'RandomForest' not in mean_squared:
                mean_squared['RandomForest'] = round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] = round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
            else:
                mean_squared['RandomForest'] += round(sm.mean_squared_error(y_test,rf.predict(X_test)),2)
                explained_variance['RandomForest'] += round(sm.explained_variance_score(y_test,rf.predict(X_test)),4)
                
                
                
            bagging_clf = BaggingRegressor(DecisionTreeRegressor(), 
                                n_estimators = 100, max_samples = 30,
                                bootstrap = True, oob_score = True)
            bagging_clf.fit(X_train,y_train)
            
             
            if 'Bagging' not in mean_squared:
                mean_squared['Bagging'] = round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] = round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)
            else:
                mean_squared['Bagging'] += round(sm.mean_squared_error(y_test,bagging_clf.predict(X_test)),2)
                explained_variance['Bagging'] += round(sm.explained_variance_score(y_test,bagging_clf.predict(X_test)),4)

           

                
                
            d_clf = DecisionTreeRegressor()
            d_clf.fit(X_train, y_train)
            
            if 'Decision' not in mean_squared:
                mean_squared['Decision'] = round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] = round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)
            else:
                mean_squared['Decision'] += round(sm.mean_squared_error(y_test,d_clf.predict(X_test)),2)
                explained_variance['Decision'] += round(sm.explained_variance_score(y_test,d_clf.predict(X_test)),4)

                
                
                
            k_clf = KNeighborsRegressor(4)
            k_clf.fit(X_train, y_train)
            
            if 'KNN' not in mean_squared:
                mean_squared['KNN'] = round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] = round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)
            else:
                mean_squared['KNN'] += round(sm.mean_squared_error(y_test,k_clf.predict(X_test)),2)
                explained_variance['KNN'] += round(sm.explained_variance_score(y_test,k_clf.predict(X_test)),4)

                
    print(k)
    for i in mean_squared:
        mean_squared[i] = round(mean_squared[i]/40,2)
        explained_variance[i] = round(explained_variance[i]/40,4)
    
    for i in range(len(Al)):
        sheet.write(k+1, i+1, mean_squared[Al[i]])
        sheet.write(k+9, i+1, explained_variance[Al[i]])
        
    
    print('mean_squared_error:', mean_squared)
    print()
    print('explained_variance_score', explained_variance)
    print()
    
wbk.close()