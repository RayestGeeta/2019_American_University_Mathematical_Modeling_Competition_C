import matplotlib.pyplot as plt

path = '/Users/zhenshihao/Desktop/2018_MCMProblemC_DATA/Useful-Data/1369_/'
os.chdir(path)
plt.figure(figsize = (10,8))

Df1 = pd.read_excel('All_Small_feature.xlsx')
Df2 = pd.read_excel('Small_Some_feature.xlsx')
Df3 = pd.read_excel('Special_Some_feature.xlsx')
Df1.drop(columns=['mean_squared_error'])
Df2.drop(columns=['mean_squared_error'])
Df3.drop(columns=['mean_squared_error'])
year = [i for i in range(2010,2017)]
for i in Al:
    d1 = Df1[i][:7]
    d2 = Df2[i][:7]
    d3 = Df3[i][:7]
    plt.plot(year, d1-d2, label = i)
    plt.plot(year, d1-d3, label = i, linestyle = '--')
    
plt.legend(loc = 2)
plt.title('All-selected-special')
plt.xlabel('year')
plt.ylabel('mean_squared_error')
plt.savefig('M_Small_All_selected_special.jpg')
plt.show()

plt.figure(figsize = (10,8))
for i in Al:
    d1 = Df1[i][8:15]
    d2 = Df2[i][8:15]
    d3 = Df3[i][8:15]
    plt.plot(year, d2-d1, label = i)
    plt.plot(year, d3-d1, label = i, linestyle = '--')
    
plt.legend(loc = 2)
plt.title('All-selected-special')
plt.xlabel('year')
plt.ylabel('explained_variance_score')
plt.savefig('E_Small_All_selected_special.jpg')
plt.show()




# year = [i for i in range(2010,2017)]
# for i in Al:
#     d1 = Df1[i][:7]
    
    
#     plt.plot(year, d1-d3+10000, label = i)
    
# plt.legend()
# plt.title('All-special')
# plt.xlabel('year')
# plt.ylabel('mean_squared_error')
# plt.savefig('All_special.jpg')
# plt.show()


path = '/Users/zhenshihao/Desktop/2018_MCMProblemC_DATA/Useful-Data'
os.chdir(path)
plt.figure(figsize = (10,8))
Df1 = pd.read_excel('All_feature.xlsx')
Df2 = pd.read_excel('Some_feature.xlsx')
Df1.drop(columns=['mean_squared_error'])
Df2.drop(columns=['mean_squared_error'])
Df3 = pd.read_excel('Special_feature.xlsx')
Df3.drop(columns=['mean_squared_error'])
year = [i for i in range(2010,2017)]
for i in Al:
    d1 = Df1[i][:7]
    d2 = Df2[i][:7]
    d3 = Df3[i][:7]
    plt.plot(year, d1-d2, label = i)
    plt.plot(year, d1-d3+10000, label = i, linestyle = '--')
    
plt.legend(loc = 2)
plt.title('All-selected-special')
plt.xlabel('year')
plt.ylabel('mean_squared_error')
plt.savefig('M_All_selected_special.jpg')
plt.show()

plt.figure(figsize = (10,8))
for i in Al:
    d1 = Df1[i][8:15]
    d2 = Df2[i][8:15]
    d3 = Df3[i][8:15]
    plt.plot(year, d2-d1, label = i)
    plt.plot(year, d3-d1, label = i, linestyle = '--')
    
plt.legend(loc = 1)
plt.title('All-selected-special')
plt.xlabel('year')
plt.ylabel('explained_variance_score')
plt.savefig('E_All_selected_special.jpg')
plt.show()

