#!/usr/bin/env python
# coding: utf-8

# In[25]:


def to_float(infile, outfile):
    import pandas as pd
    df= pd.read_csv(infile)
    df['sepal.length'] = df['sepal.length'].astype(float)
    df['sepal.width'] = df['sepal.width'].astype(float)
    df['petal.length'] = df['petal.length'].astype(float)
    df['petal.width'] = df['petal.width'].astype(float)
    df.to_csv(outfile,index = False, header=False)
    #print(df)


# In[26]:


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    import math
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)


# In[27]:


def get_distances(test_row, csv_reader):
    distances=[]
    for row in csv_reader:
        row[0]=float(row[0])
        row[1]=float(row[1])
        row[2]=float(row[2])
        row[3]=float(row[3])
        #print(row[:-1])        
        #print(euclidean_distance(test_row, row[:-1]))
        distances.append(euclidean_distance(test_row, row[:-1]))        
    #distances.pop()
    return distances


# In[ ]:





# In[28]:


def save_df_with_distances_in_file(df, file):
    import pandas as pd
    df.to_csv(file,header=['sepal.length','sepal.width','petal.length','petal.length','variety','distance'],index = False)


# In[29]:


def find_nearest_neigbour(df,num_neighbors):
    #df= pd.read_csv('iris-v5.csv')
    #df.shape
    df1=df.nsmallest(num_neighbors, 'distance', keep='first')
    df1.columns=['sepal.length','sepal.width','petal.length','petal.length','variety','distance']    
    return df1


# In[30]:


def KNN(test_row, file_name, number_of_neigbours):
    import csv
    to_float(file_name,'iris-float.csv')

    file = open("iris-float.csv", newline='')
    csv_reader = csv.reader(file)

    distances=get_distances(test_row, csv_reader)
    print("distances= ",len(distances))
    df['distance']=distances
    
    df1=find_nearest_neigbour(df,number_of_neigbours)
    
    print(df1) 
    count_series=df1.variety.value_counts()
    print(count_series)    
    result=count_series.idxmax()
    return result


# In[31]:


####################################################################################
#################### get One random test case from CSV file ########################
####################################################################################
def get_random_test_case_from_file(test_cases_file_name):
    import random as rand
    import csv
    rand_index=rand.randint(1,151)
    
    to_float(test_cases_file_name,'iris-float-test-cases.csv')
    csv_reader = csv.reader('iris-float-test-cases.csv')
    
    test_cases_float_file = open("iris-float-test-cases.csv", newline='')
    csv_reader = csv.reader(test_cases_float_file)
    list=[]
    for row in csv_reader:
        row[0]=float(row[0])
        row[1]=float(row[1])
        row[2]=float(row[2])
        row[3]=float(row[3])        
        #print(row[:4])    
        list.append(row[:4])
    print(rand_index+2)
    print(list[rand_index])
    return list[rand_index]


# In[32]:


####################################################################################
########################## read one test cases from  user ###############################
####################################################################################
def read_test_case_from_user():
    test_case=[]
    test_case.append(float(input("sepal length: ")))
    test_case.append(float(input("sepal width: ")))
    test_case.append(float(input("Petal length: ")))
    test_case.append(float(input("petal width:" )))
    print(test_case)
    return test_case


# In[33]:


####################################################################################
################### get test case from a list of fixed test cases ##################
####################################################################################
def get_test_case_from_list(num_neigbours, data_file):
    test_cases_list=[[1, 2, 3, 4], [3, 3, 2, 1], [1, 1, 3, 2], [2, 2, 1, 4], [5, 6, 7, 8]]
    for test_row in test_cases_list:
        result=KNN(test_row,data_file, num_neigbours)
        print("Predicted class= ",result)


# In[34]:


####################################################################################
########################## One fixed test case #####################################
####################################################################################
def get_fixed_test_case():
    test_row=[5.8, 2.7, 5.1, 5.1]
    return test_row
    


# In[ ]:


def menu(file_name, num_neighbors):
        while True:
            print("======================================================",end="\n")
            print("          Please select one choice from Menu          ",end="\n")
            print("======================================================",end="\n")
            print("1- Test KNN using preddefined test case",end="\n")
            print("2- Test KNN using from a list of preddefined test case",end="\n")
            print("3- Test KNN using random test case from test file",end="\n")
            print("4- Test KNN using a user provided test case",end="\n")
            print("5- exit ",end="\n")
            print("======================================================",end="\n")
            choice=int(input("Enter your choice: "))
    
            if (choice==1):
                test_row=[]
                test_row=get_fixed_test_case()                
                num_neighbors=int(input("Enter Number of Neigbours (K): "))
                result=KNN(test_row,file_name, num_neighbors)
                print("Predicted class= ",result)
            
            elif (choice==2):
                num_neighbors=int(input("Enter Number of Neigbours (K): "))
                test_row=[]
                test_row=get_test_case_from_list(num_neighbors, file_name)            
                #KNN(test_row,file_name, num_neighbors)

            elif (choice==3):
                test_cases_file_name="iris-test-cases-v2.csv"            
                num_neighbors=int(input("Enter Number of Neigbours (K): "))
                test_row=[]
                test_row= get_random_test_case_from_file(test_cases_file_name)
                result=KNN(test_row,file_name, num_neighbors)
                print("Predicted class= ",result)

            elif (choice==4):
                num_neighbors=int(input("Enter Number of Neigbours (K): "))
                test_row=[]
                test_row=read_test_case_from_user()
                result=KNN(test_row,file_name, num_neighbors)
                print("Predicted class= ",result)
            else:
                break
            


# In[38]:


def main():
    import numpy as np
    import pandas as pd
    file_name='iris.csv'
    df= pd.read_csv(file_name)
    num_neighbors=5;
    menu(file_name, num_neighbors)
    #test_row=get_fixed_test_case()
    #test_cases_file="iris-float-test-cases.csv"
    #test_row= get_random_test_case_from_file(test_cases_file)
main()


# In[ ]:




