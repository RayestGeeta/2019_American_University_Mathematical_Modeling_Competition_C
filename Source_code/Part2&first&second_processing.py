county.remove(51515)



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
        
        if i >= 2013:
            Df = Df.drop(columns = ['HC01_VC217'])
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
        print('2'+string+' end!')