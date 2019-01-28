import os
path = '/Users/zhenshihao/Desktop/2018_MCMProblemC_DATA/Useful-Data/1369_/'
os.chdir(path)

m = '1369_'


for i in Opioid_country:
    if i!=2017:
        array = list(0 for i in range(len(county)))
        string = 'eco1' + str(i)[-1] + '.csv'
        Df = pd.read_csv(string)
        Df.index = Df['GEO.id2']
        Df = Df.drop(columns=['GEO.id2'])
        if 51515 in Df.index:
            Df = Df.drop(index = [51159,51161,51685,51515])
        else:
            Df = Df.drop(index = [51159,51161,51685])

#             if i >= 2013:
#                 Df = Df.drop(columns = ['HC01_VC217'])
        Dict_county = {}
        for j in Opioid_country[i]:
            if j not in Dict_county:
                Dict_county[j] = Opioid_country[i][j][0]['Total_Reports']
            else:
                Dict_county[j] += Opioid_country[i][j][0]['Total_Reports']


        if 51515 in Dict_county:
            Dict_county.pop(51515)
        for i in Dict_county:
            array[county.index(i)] = Dict_county[i]

        array = np.array(array)

        Df['TotalDrugReports'] = array
        Df.to_csv('2'+string)
        print(m + '_2'+string+' end!'+str(len(Df.columns)))

warnings.filterwarnings("ignore")


scores = {}
mdi_scores = {}
mda_scores = {}
ss_scores = {}

wbk = xlsxwriter.Workbook(m + '_feature_correlation.xlsx')
sheet = wbk.add_worksheet('Data')

sheet.write(0, 0,'R2')
sheet.write(8, 0, 'MDI')
sheet.write(16, 0, 'MDA')
sheet.write(24, 0, 'SS')

sheet.write(33, 0, 'R2')
sheet.write(34, 0, 'MDI')
sheet.write(35, 0, 'MDA')
sheet.write(36, 0, 'SS')


for i in range(0, 74):
    sheet.write(0, i+1, i)

for i in range(0, 32, 8):
    for j in range(2010, 2017):
        sheet.write(i+j%2009, 0, j)




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

    scores_sum = [i for i in range(0, len(Df.columns[1:-1]))]
    mdi_scores_sum = [i for i in range(0, len(Df.columns[1:-1]))]
    mda_scores_sum = [i for i in range(0, len(Df.columns[1:-1]))]
    ss_scores_sum = [i for i in range(0, len(Df.columns[1:-1]))]

    for m in range(10):
        X = np.array(datas)

        Y = np.array(label)



        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), 3, .3))
            if names[i] in scores:
                scores[names[i]] += round(np.mean(score), 4)
            else:
                scores[names[i]] = round(np.mean(score), 4)


            scores_sum[i] += round(np.mean(score), 4)
#             print('R2:',String, 'end.', len(scores) )
            if(m == 9):
                sheet.write(k+1, i+1, scores_sum[i]/10)
    #         sheet.write(k+1, i+1, round(np.mean(score), 4))

    
        rf = RandomForestRegressor(n_estimators=20, max_depth=4)
        for i in range(X.shape[1]):
            score = cross_val_score(rf, X[:, i:i+1], Y, scoring="r2",cv=ShuffleSplit(len(X), 3, .3))
            if names[i] in scores:
                scores[names[i]] += round(np.mean(score), 4)
            else:
                scores[names[i]] = round(np.mean(score), 4)
    #     print('R2:',String, 'end.', len(scores) )
            scores_sum[i] += round(np.mean(score), 4)
            if(m == 9):
                sheet.write(k+1, i+1, scores_sum[i]/10)


        rf = RandomForestRegressor()
        rf.fit(X, Y)
        mdi_score = list(map(lambda x: round(x, 4), rf.feature_importances_))
        for i in range(len(mdi_score)):
            if i in mdi_scores:
                mdi_scores[i] += mdi_score[i]
            else:
                mdi_scores[i] = mdi_score[i]

            mdi_scores_sum[i] += mdi_score[i]
            if(m == 9):
                sheet.write(9+k, i+1, mdi_scores_sum[i]/10)
        print('MDI', String, 'end.', len(mdi_scores))
    #         sheet.write(9+k, i+1, mdi_score[i])


        rf = RandomForestRegressor()
        mda_score = defaultdict(list)
        for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            r = rf.fit(X_train, Y_train)
            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(X.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                mda_score[names[i]].append((acc-shuff_acc)/acc)
        mda_score = [[round(np.mean(score), 4), feat] for feat,score in mda_score.items()]
        for i in range(0,len(mda_score)):
            if i in mda_scores:
                mda_scores[i] += mda_score[i][0]
            else:
                mda_scores[i] = mda_score[i][0]

            mda_scores_sum[i] += mda_score[i][0]
            if(m == 9):
                sheet.write(17+k, i+1, mda_scores_sum[i]/10)
        print('MDA', String, 'end.', len(mda_scores))
    #         sheet.write(17+k, i+1, mda_score[i][0])


        rlasso = RandomizedLasso(alpha=0.025)
        rlasso.fit(X, Y)
        ss_score = list(map(lambda x: round(x, 4), rlasso.scores_))
        for i in range(len(ss_score)):
            if i in ss_scores:
                ss_scores[i] += ss_score[i]
            else:
                ss_scores[i] = ss_score[i]

            ss_scores_sum[i] += ss_score[i]
            if(m == 9):
                sheet.write(25+k, i+1, ss_scores_sum[i]/10)
        print('SS', String, 'end', len(ss_scores))
        print()
    #         sheet.write(25+k, i+1, ss_score[i])




for i in scores:
    scores[i] = round(scores[i]/70,4)
    sheet.write(33, i+1, scores[i])

for i in mdi_scores:
    mdi_scores[i] = round(mdi_scores[i]/70, 4)
    sheet.write(34, i+1, mdi_scores[i])

for i in mda_scores:
    mda_scores[i] = round(mda_scores[i]/70, 4)
    sheet.write(35, i+1, mda_scores[i])


for i in ss_scores:
    ss_scores[i] = round(ss_scores[i]/70, 4)
    sheet.write(36, i+1, ss_scores[i])


print('R2: ',scores)
print()
print('MDI:',mdi_scores)
print()
print('MDA:',mda_scores)
print()
print('SS:',ss_scores)

wbk.close()