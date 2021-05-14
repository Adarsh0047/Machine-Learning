from pandas import *
df=read_csv('D:\\Old pc\\Python\\Python Lab\\AIML\\names_sample.csv')

print(df ,"\n \n \n")
df.sort_values('Male',axis=0,ascending=True,inplace=True,na_position='last')
print(df)

