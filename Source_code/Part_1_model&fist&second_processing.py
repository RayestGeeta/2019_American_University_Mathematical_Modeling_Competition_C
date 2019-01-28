import xlrd
import xlsxwriter
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
import matplotlib.pyplot as pyplot

wb = xlrd.open_workbook(filename='MCM_NFLIS_Data.xlsx')
sheet = wb.sheet_by_name('Data')

Datas = []
ncols = sheet.ncols
nrows = sheet.nrows

State_code = {}
County_code = {}
for i in range(1,nrows):
    data = sheet.row_values(i)
    
    if data[3] not in State_code:
        State_code[data[3]] = data[1]
        
    if data[4] not in State_code:
        County_code[data[4]] = data[2]
    for j in range(4):
        data.pop(1)
    data[0] = int(data[0])
    data[1] = int(data[1])
    Datas.append(data)
    
county = []
opioid = []
for i in Datas:
    if i[1] not in county:
        county.append(i[1])

county.sort()

for i in Datas:
    if i[2] not in opioid:
        opioid.append(i[2])
        
        
        
        
Opioid_country = {}

for i in Datas:
    if i[0] not in Opioid_country:
        opioid_country = {}
        if i[1] not in opioid_country:
            opioid_country[i[1]] = []
            opioid_country[i[1]].append({'Total':i[4]})
            new_opioid = {}
            new_opioid[i[2]] = i[3]
            opioid_country[i[1]].append(new_opioid)
        
        else:
            new_opioid = {}
            new_opioid[i[2]] = i[3]
            opioid_country[i[1]].append(new_opioid)
        
        Opioid_country[i[0]] = opioid_country
        
    else:
        if i[1] not in Opioid_country[i[0]]:
            Opioid_country[i[0]][i[1]] = []
            Opioid_country[i[0]][i[1]].append({'Total':i[4]})
            new_opioid = {}
            new_opioid[i[2]] = i[3]
            Opioid_country[i[0]][i[1]].append(new_opioid)
            
        else:
            new_opioid = {}
            new_opioid[i[2]] = i[3]
            Opioid_country[i[0]][i[1]].append(new_opioid)
            
for i in Opioid_country:
    for j in Opioid_country[i]:
        sum = 0.0
        for k in Opioid_country[i][j]:
            for m in k:
                sum += k[m]
        Opioid_country[i][j].insert(0,{'Total_Reports':sum - Opioid_country[i][j][0]['Total']})
        
        
        


wbk = xlsxwriter.Workbook('Opioid_county.xlsx')
sheet = wbk.add_worksheet('Data')

sheet.write(0, 0, 'YYYY')
sheet.write(0, 1, 'FIPS_Combined')
sheet.write(0, 2, 'SubstanceName')
sheet.write(0, 3, 'DrugReports')
sheet.write(0, 4, 'TrueTotalDrugReports')
sheet.write(0, 5, 'TotalDrugReportsCounty')

count = 1
for i in Opioid_country:
    for j in Opioid_country[i]:
        for k in range(2, len(Opioid_country[i][j])):
            sheet.write(count, 0, i)
            sheet.write(count, 1, j)
            for m in Opioid_country[i][j][k]:
                sheet.write(count, 2, m)
                sheet.write(count, 3, Opioid_country[i][j][k][m])
                sheet.write(count, 4, Opioid_country[i][j][0]['Total_Reports'])
                sheet.write(count, 5, Opioid_country[i][j][1]['Total'])
                count += 1
                
wbk.close()


Opioid_sub = {}

for i in Datas:
    if i[0] not in Opioid_sub:
        opioid_sub = {}
        if i[2] not in opioid_sub:
            opioid_sub[i[2]] = []
            new = {}
            new[i[1]] = i[3]
            opioid_sub[i[2]].append(new)
            
        else:
            new = {}
            new[i[1]] = i[3]
            opioid_sub[i[2]].append(new)
            
        Opioid_sub[i[0]] = opioid_sub
        
    else:
        if i[2] not in Opioid_sub[i[0]]:
            Opioid_sub[i[0]][i[2]] = []
            new_opioid = {}
            new_opioid[i[1]] = i[3]
            Opioid_sub[i[0]][i[2]].append(new_opioid)
            
        else:
            new_opioid = {}
            new_opioid[i[1]] = i[3]
            Opioid_sub[i[0]][i[2]].append(new_opioid)

for i in Opioid_sub:
    for j in Opioid_sub[i]:
        sum = 0.0
        for k in Opioid_sub[i][j]:
            for m in k:
                sum += k[m]
        Opioid_sub[i][j].insert(0,{'Total_Reports':sum})
        Opioid_sub[i][j].insert(0,{'Total_County':len(Opioid_sub[i][j])-1})
        
wbk = xlsxwriter.Workbook('Opioid_sub.xlsx')
sheet = wbk.add_worksheet('Data')

sheet.write(0, 0, 'YYYY')
sheet.write(0, 1, 'SubstanceName')
sheet.write(0, 2, 'FIPS_Combined')
sheet.write(0, 3, 'DrugReports')
sheet.write(0, 4, 'TotalCountReports')
sheet.write(0, 5, 'TotalCountCounty')


count = 1
for i in Opioid_sub:
    for j in Opioid_sub[i]:
        for k in range(2, len(Opioid_sub[i][j])):
            sheet.write(count, 0, i)
            sheet.write(count, 1, j)
            for m in Opioid_sub[i][j][k]:
                sheet.write(count, 2, m)
                sheet.write(count, 3, Opioid_sub[i][j][k][m])
                sheet.write(count, 4, Opioid_sub[i][j][1]['Total_Reports'])
                sheet.write(count, 5, Opioid_sub[i][j][0]['Total_County'])
                count += 1
                
wbk.close()



wbk = xlsxwriter.Workbook('Opioid_spread.xlsx')
sheet = wbk.add_worksheet('Data')

for i in range(1,len(county)+1):
    sheet.write(0, i, county[i-1])

sheet.write(0, len(county)+1, 'Total')
for i in range(0,9*len(opioid),9):
    sheet.write(i, 0, opioid[int(i/9)])
    sheet.write(i+1, 0, 2010)
    sheet.write(i+2, 0, 2011)
    sheet.write(i+3, 0, 2012)
    sheet.write(i+4, 0, 2013)
    sheet.write(i+5, 0, 2014)
    sheet.write(i+6, 0, 2015)
    sheet.write(i+7, 0, 2016)
    sheet.write(i+8, 0, 2017)

for i in range(2010,2018):
    for j in Opioid_country[i]:
        y = county.index(j) + 1
        for k in range(2,len(Opioid_country[i][j])):
            for m in Opioid_country[i][j][k]:
                x = opioid.index(m)*9 + i%2009
                
                sheet.write(x, y, Opioid_country[i][j][k][m])
wbk.close()




Dict_county = {}
for i in Opioid_country:
    for j in Opioid_country[i]:
        if j not in Dict_county:
            Dict_county[j] = Opioid_country[i][j][0]['Total_Reports']
        else:
            Dict_county[j] += Opioid_country[i][j][0]['Total_Reports']

Dict_True = {}
for i in Dict_county:
    if Dict_county[i] > 1000:
        Dict_True[i] = Dict_county[i]
        
True_county = []
for i in Dict_True:
    True_county.append(i)

True_county.sort()
    
Dict_opioid = {}
for i in Opioid_sub:
    for j in Opioid_sub[i]:
        if j not in Dict_opioid:
            Dict_opioid[j] = Opioid_sub[i][j][1]['Total_Reports']
            
        else:
            Dict_opioid[j] += Opioid_sub[i][j][1]['Total_Reports']
            
Dict_True_2 = {}
for i in Dict_opioid:
    if Dict_opioid[i] > 900:
        Dict_True_2[i] = Dict_opioid[i]

True_opioid = []
for i in Dict_True_2:
    True_opioid.append(i)

for i in range(5):
    True_opioid.pop()
wbk = xlsxwriter.Workbook('Opioid_spread_True.xlsx')
sheet = wbk.add_worksheet('Data')

for i in range(1,len(True_county)+1):
    sheet.write(0, i, True_county[i-1])

sheet.write(0, len(True_county)+1, 'Total')

for i in range(0,9*len(True_opioid),9):
    sheet.write(i, 0, True_opioid[int(i/9)])
    sheet.write(i+1, 0, 2010)
    sheet.write(i+2, 0, 2011)
    sheet.write(i+3, 0, 2012)
    sheet.write(i+4, 0, 2013)
    sheet.write(i+5, 0, 2014)
    sheet.write(i+6, 0, 2015)
    sheet.write(i+7, 0, 2016)
    sheet.write(i+8, 0, 2017)

for i in range(2010,2018):
    for j in Opioid_country[i]:
        if j in True_county:
            y = True_county.index(j) + 1
            for k in range(2,len(Opioid_country[i][j])):
                for m in Opioid_country[i][j][k]:
                    if m in True_opioid:
                        x = True_opioid.index(m)*9 + i%2009
                        sheet.write(x, y, Opioid_country[i][j][k][m])
wbk.close()




False_county = []
False_opioid = []

for i in county:
    if i not in True_county:
        False_county.append(i)


for i in opioid:
    if i not in True_opioid:
        False_opioid.append(i)
        
        
wbk = xlsxwriter.Workbook('Opioid_spread_False.xlsx')
sheet = wbk.add_worksheet('Data')

for i in range(1,len(False_county)+1):
    sheet.write(0, i, False_county[i-1])

sheet.write(0, len(False_county)+1, 'Total')

for i in range(0,9*len(False_opioid),9):
    sheet.write(i, 0, False_opioid[int(i/9)])
    sheet.write(i+1, 0, 2010)
    sheet.write(i+2, 0, 2011)
    sheet.write(i+3, 0, 2012)
    sheet.write(i+4, 0, 2013)
    sheet.write(i+5, 0, 2014)
    sheet.write(i+6, 0, 2015)
    sheet.write(i+7, 0, 2016)
    sheet.write(i+8, 0, 2017)

for i in range(2010,2018):
    for j in Opioid_country[i]:
        if j in False_county:
            y = False_county.index(j) + 1
            for k in range(2,len(Opioid_country[i][j])):
                for m in Opioid_country[i][j][k]:
                    if m in False_opioid:
                        x = False_opioid.index(m)*9 + i%2009
                        sheet.write(x, y, Opioid_country[i][j][k][m])
wbk.close()

print('End')

count = 0
wbk = xlsxwriter.Workbook('Opioid_spread_Result.xlsx')
sheet = wbk.add_worksheet('Data')
Model = []
for i in range(1, len(True_county)+1):
    sheet.write(0, i, True_county[i-1])
D = pd.read_excel('Opioid_spread_True.xlsx')
R = pd.DataFrame(columns=D.columns)
for k in range(len(True_opioid)):
   
    
    D1 = D[k*9:k*9+8].fillna(0)
    D1.index =pd.date_range('2010',periods=8, freq='Y')
    D1 = D1.drop(columns=['Morphine','Total'])
    
    Drop = {}
    for i in D1.columns:
        for j in range(1,len(D1[i])-1):
            if(D1[i][j]!=D1[i][j-1]):
                break
        if j==len(D1[i])-2:
            Drop[i] = D1[i][j]
            D1 = D1.drop(columns=i)
        
    train = D1[:int(0.9*(len(D1)))]
    valid = D1[int(0.9*(len(D1))):]

    model = VAR(endog = train)
    model_fit = model.fit()
    prediction = model_fit.forecast(model_fit.y, steps=len(valid))

    pred = pd.DataFrame(index=range(0,len(prediction)),columns=[train.columns])
    for j in range(0,len(pred.columns)):
        for i in range(0, len(prediction)):
            pred.iloc[i][j] = prediction[i][j]
    
    error = 0
    for i in train.columns:
        error += math.sqrt(mean_squared_error(pred[i], valid[i]))
    error = error/len(train.columns)
    print('error for',True_opioid[k],':',error)
    
    model = VAR(endog=D1)
    model_fit = model.fit()
    result = model_fit.forecast(model_fit.y, steps=1)[0]
    
    result = result.tolist()

    for i in Drop:
        result.insert(True_county.index(i), Drop[i])
    
    for i in range(0,len(result)):
        if result[i] < 0.0:
            result[i] = 0
        else:
            result[i] = int(result[i] + 0.5)
    result = np.array(result)
    sheet.write(k+1, 0, True_opioid[k])
    for i in range(len(result)):
        sheet.write(k+1, i+1, result[i])
    print(result)
    Model.append(model_fit)
#     if(len(Model)==2):
#         print(model)
wbk.close()