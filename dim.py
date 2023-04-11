import pandas as pd

df = pd.read_excel("5.xlsx")
df1=df.iloc[:,0]
df2 = df.iloc[:,1]

var1=df1.var()
var2=df2.var()

list1=list(df1)
list2=list(df2)
list3=[]
disall=0
for i in range(0, len(list1)):
    for j in range(i+1,len(list1)):
        dis=pow(list1[i]-list1[j],2)+pow(list2[i]-list2[j],2)
        list3.append(dis)
        disall+=dis
d=disall/len(list1)
df3=pd.DataFrame(list3)
m=df3.var()
dm=d/2/m
print(dm)

