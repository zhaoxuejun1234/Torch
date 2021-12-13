import os
import sys
import numpy as np
a = float("123")
b = list()
b =[]
c = dict()
c = {}


a ="abc"
a= "age is %d" %15
age = 150
a=  f"age is {age}"
print(a)
t=(12,12)


a=True
while a:
    print("sdsds")
    break

a=[1,2,3,4,5,5]
for index,value in enumerate(a):
    print(f"{index},{value}")



a = [item+2 for item in a if item>2]
print(a)



isok = True
a = 1 if isok else 2
print(a)
if a==5:
    print("aaa")
elif a==3:
    print("cccc")



with open("a.txt","w") as f:
    f.write("a")



a= [12,32,42,22,333,2131,22,42]
print(list(set(a)))
