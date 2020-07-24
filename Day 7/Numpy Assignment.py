# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:08:56 2020

@author: amitdharamsi
"""
import numpy as np

#version_number
print(np.__version__)
print("\n")

#1D Array
array1 = np.arange(10)
print(array1)
print("\n")

#2D array
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)
print("\n")

b = np.array( [ (1.5,2,4) , (4,5,6) ] ,dtype=float )
print(b)
print("\n")

#Boolean array
array2 = np.full((4,3), True, dtype=bool)
print(array2)
print("\n")

#Arrange
#creates an array of evenly spaced values
#last value is not included
d = np.arange(0,10)
print(d)
print("\n")

d1 = np.arange(0,10,3)
print(d1)
print("\n")

#ZEROES
#create an arrays of zeros
# 1D array
a = np.zeros(5,dtype=int)
print(a)
print("\n")

#2D array
a = np.zeros((2,5),dtype=int)
print(a)
print("\n")

#ONES
#create an arrays of ones
#1D
a = np.ones(5)
print(a)
print("\n")

#2D
a = np.ones((2,5),dtype=np.int8)
print(a)
print("\n")

#EYE
a = np.eye(4)
print(a)
print("\n")

#LINSPACE
#create an array of evenly space values
a = np.linspace(0,50,11,dtype=int)
print(a)
print("\n")

a = np.linspace(0,5,18,dtype=float)
print(a)
print("\n")


#RANDOM
#create an array with random values
#1D
print("RANDOM")
a = np.random.random(5)
print(a)
print("\n")

#2D
a =np.random.rand(2,3)
print(a,"\n")

#another way
a1 = np.random.random((2,3))
print(a1,"\n")

#By using randint =>> this will print any random no between 0 to 3 (3 included)
rand1 = np.random.randint(3)
print(rand1,'\n')

rand2 = np.random.randint(1,10) #10 excluded
print(rand2,'\n')

#If we want to insert random integer to array then we can do like below
my_arr = np.random.randint(1,10,6)
print(my_arr,'\n')

#MAX and MIN

max = my_arr.max()
min = my_arr.min()
print("Max : ", max)
print("Min : ", min,'\n')

#To return the index of max and min element
max_index = my_arr.argmax()
min_index = my_arr.argmin()

print("Max_Index :",max_index)
print("Max_Index :",min_index,'\n')

#Reshaping
#convert 1d array to 2D
# [6 3 1 6 5 2] 
new_arr = my_arr.reshape(2,3)
print(new_arr)
print('\n')

#Creating Copy
my_arr1 = my_arr.copy()
print(my_arr1)
print('\n')

#2D Arrays
array2d = np.array( [ (1,2,3) , (4,5,6) , (6,78,7) ] ) 
print(array2d,'\n')

#access a element
print(array2d[1][1],'\n')

#another way
print(array2d[1,1],'\n')


#SLICING

print(array2d,'\n')

print(array2d[1:])
print('\n')

print(array2d[1:,1:])
print('\n')

print(array2d[0:2])
print('\n')

print(array2d[::-1])
print('\n')


#COMPARISON
print("Comparison")
print(my_arr)
print('\n')

#Return true and false 
arr = my_arr>2
print(arr)
print('\n')

#returns the value 
arr1 = my_arr[my_arr>2]
print(arr1)
print('\n')

#OPERATIONS
print("Operations")
print(my_arr)
print('\n')
print(my_arr1)
print('\n')

#Addition
print("Addition")
add_two_arr = my_arr + my_arr1
print(add_two_arr)
print('\n')

#Subtraction
print("Subtraction")
sub_two_arr = my_arr - my_arr1
print(sub_two_arr)
print('\n')

#Multiplication
print("multiplication")
Mul_two_arr = my_arr * my_arr1
print(Mul_two_arr)
print('\n')

#Divide
print("divide")
div_two_arr = my_arr / my_arr1
print(div_two_arr)
print('\n')

#add 10 to all
a = my_arr+10
print(a)
print('\n')
#Similar for m,d,s

#Square root
sq_arr = np.sqrt(my_arr)
print(sq_arr)
print('\n')

#Aggregate
print("Aggregate Functions")
print(my_arr)
print('\n')

#Sum
sum_arr = my_arr.sum()
print(sum_arr)
print('\n')

#Min
min_arr = my_arr.min()
print(min_arr)
print('\n')

#Max
max_arr = my_arr.max()
print(max_arr)
print('\n')

#mean
mean_arr = my_arr.mean()
print(mean_arr)
print('\n')


#Create a 3×3 numpy array of all True’s
print("Create a 3×3 numpy array of all True’s")
a = np.full((3, 3), True, dtype=bool)
print(a)
print('\n')


#Extract all odd numbers from arr
print("Extract all odd numbers from arr")
a = np.arange(1,11)
print(a)
a = a[a%2 == 1]
print(a)
print('\n')


#Replace all odd numbers in arr with -1
print("Replace all odd numbers in arr with -1")
a = np.arange(1,11)
print(a)
a[a%2 == 1] = -1
print(a)
print('\n')

#Replace all odd numbers in arr with -1 without changing arr
print("Replace all odd numbers in arr with -1 without changing arr")
arr = np.arange(11)
print(arr)
out = np.where(arr % 2 == 1, -1, arr)
print("arr:", arr)
print("out: ",out)
print('\n')

#Convert a 1D array to a 2D array with 2 rows
print("Convert a 1D array to a 2D array with 2 rows")
arr = np.arange(10)
a = arr.reshape(2,-1)  # Setting to -1 automatically decides the number of cols
print(arr)
print(a)
print('\n')

#Stack arrays a and b vertically
print("Stack arrays a and b vertically")
a = np.arange(10).reshape(2,-1)
b = np.repeat(3, 10).reshape(2,-1)
print("a: ",a)
print("b: ",b)
arr = np.vstack((a,b))
print("arr: ",arr)
print('\n')

#Stack the arrays a and b horizontally.
print("Stack the arrays a and b horizontally.")
a = np.arange(10).reshape(2,-1)
b = np.repeat(3, 10).reshape(2,-1)
print("a: ",a)
print("b: ",b)
arr = np.hstack((a,b))
print("arr: ",arr)
print('\n')

#Create the following pattern without hardcoding. Use only numpy functions and the below input array a.
a = np.array([1,2,3])
arr = np.r_[np.repeat(a, 3), np.tile(a, 3)]
print("Create Pattern")
print("a =", a)
print("arr =", arr)
print('\n')

#Get the common items between a and b
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("common items in a and b")
c = np.intersect1d(a,b)
print(c)
print('\n')


#From array a remove all items present in array b
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
c = np.setdiff1d(a,b)
print("From array a remove all items present in array b")
print(c)
print('\n')

#Get the positions where elements of a and b match
print('Get the positions where elements of a and b match')
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
c = np.where(a == b)
print(c)
print('\n')

#How to extract all numbers between a given range from a numpy array?
print("How to extract all numbers between a given range from a numpy array?")
a = np.array([2, 6, 1, 9, 10, 3, 27])
index = np.where((a>=5) & (a<=10)) # gives the index
print(a[index]) #1st method
print(a[(a>=5)&(a<=10)])    #2nd method
print('\n')

#Swap columns 1 and 2 in the array arr
print("Swap columns 1 and 2 in the array arr")
arr = np.arange(9).reshape(3,3)
print("Original Array")
print(arr)
print("After swap")
arr = arr[:,[0,2,1]]
print(arr)
print('\n')


#Swap rows 1 and 2 in the array arr
print("Swap rows 1 and 2 in the array arr:")
arr = np.arange(9).reshape(3,3)
print("Original Array")
print(arr)
print("After swap")
arr = arr[[0,2,1],:]
print(arr)
print('\n')

#Reverse the rows of a 2D array arr
print('Reverse the rows of a 2D array arr')
arr = np.arange(9).reshape(3,3)
print("Original Array")
print(arr)
print("After swap")
arr = arr[::-1]
print(arr)
print('\n')


#Reverse the columns of a 2D array arr.
print("Reverse the columns of a 2D array arr")
arr = np.arange(9).reshape(3,3)
print("Original Array")
print(arr)
print("After swap")
arr = arr[:,::-1]
print(arr)
print('\n')


#Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.
print("Create a 2D array of shape 5x3 to contain random decimal numbers between 5 and 10.")
rand_arr =np.random.randint(low=5, high=10, size=(5,3)) + np.random.random((5,3))
print(rand_arr)
print('\n')
rand_arr = np.random.uniform(5,10, size=(5,3))
print(rand_arr)
print('\n')


#Print or show only 3 decimal places of the numpy array rand_arr.
print("Print or show only 3 decimal places of the numpy array rand_arr.")
rand_arr = np.random.random((5,3))
print(rand_arr)
#np.set_printoptions(precision=3)
rand_arr[:4]
print('\n')
#np.set_printoptions(suppress=False, precision=8)

#Pretty print rand_arr by suppressing the scientific notation (like 1e10)
print("Pretty print rand_arr by suppressing the scientific notation")
#np.set_printoptions(suppress=False, precision=6)
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
print(rand_arr)
print('\n')
#np.set_printoptions(suppress=True, precision=6)
print(rand_arr)
print('\n')
#np.set_printoptions(suppress=False, precision=8)

#Import the iris dataset keeping the text intact.
print("Import the iris dataset keeping the text intact.")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
print(iris[:3])
print('\n')

#Extract the text column species from the 1D iris imported in previous question.
print("Extract the text column species from the 1D iris imported in previous question")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_1d = np.genfromtxt(url, delimiter=',', dtype=None, encoding=None)
print(iris_1d.shape)
species = np.array([row[4] for row in iris_1d])
print(species[:5])
print('\n')


#Find the mean, median, standard deviation of iris's sepallength (1st column)
print("Find the mean, median, standard deviation of iris's sepallength (1st column)")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
mu, med, sd = np.mean(sepallength), np.median(sepallength), np.std(sepallength)
print(mu, med, sd)
print('\n')

#Create a normalized form of iris's sepallength whose values range exactly between 0 and 1 so that the minimum has value 0 and maximum has value 1.
print("Create a normalized form")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
Smax, Smin = sepallength.max(), sepallength.min()
S = (sepallength - Smin)/(Smax - Smin)
print(S)
print('\n')


#Find the 5th and 95th percentile of iris's sepallength
print("Find the percentile")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
sepallength = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])
percentile = np.percentile(sepallength, q=[5,25,50,75,95,99,100])
print(percentile)
print('\n')

#nsert values at random positions in an array
print("nsert values at random positions in an array")
print("Insert np.nan values at 30 random positions in iris_2d dataset")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
i, j = np.where(iris_2d)
np.random.seed(100)
iris_2d[np.random.choice((i), 30), np.random.choice((j), 30)] = np.nan
print(iris_2d[:10])
print('\n')

#filter a numpy array based on two or more conditions
print("Filter the rows of iris_2d that has petallength (3rd column) > 1.5 and sepallength (1st column) < 5.0")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
condition = (iris_2d[:, 2] > 1.5) & (iris_2d[:, 0] < 5.0)
print(iris_2d[condition])
print('\n')


#How to find if a given array has any null values
print("Find out if iris_2d has any missing values")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
print(np.isnan(iris_2d).any())
print('\n')

#replace all missing values with 0 in a numpy array
print("Replace all ccurrences of nan with 0 in numpy array")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
iris_2d[np.isnan(iris_2d)] = 0
print(iris_2d[:10])
print('\n')

#ind the count of unique values in a numpy array
print("find the count of unique values in a numpy array")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
species = np.array([row.tolist()[4] for row in iris])
print(np.unique(species, return_counts=True))
print('\n')

#Convert a numeric to a categorical (text) array?
print("Bin the petal length (3rd) column of iris_2d to form a text array, such that if petal length is:")
print("Less than 3 --> 'small'")
print("3-5 --> 'medium'")
print("'>=5 --> 'large'")
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
# Bin petallength 
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])
# Map it to respective category
label_map = {1: 'small', 2: 'medium', 3: 'large', 4: np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]
# View
print(petal_length_cat[:20])
print('\n')


#create a new column from existing columns of a numpy array
# Input
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='object')
# Solution
# Compute volume
sepallength = iris_2d[:, 0].astype('float')
petallength = iris_2d[:, 2].astype('float')
volume = (np.pi * petallength * (sepallength**2))/3
# Introduce new dimension to match iris_2d's
volume = volume[:, np.newaxis]
# Add the new column
out = np.hstack([iris_2d, volume])
# View
print(out[:4])
print('\n')

#sort a 2D array by a column
print("sort a 2D array by a column")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
#sort by sepal length
print(iris[iris[:,0].argsort()][:20])
print('\n')

#find the most frequent value in a numpy array
print("find the most frequent value in a numpy array")
# Input:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Solution:
vals, counts = np.unique(iris[:, 2], return_counts=True)
print(vals[np.argmax(counts)])
print('\n')


#find the position of the first occurrence of a value greater than a given value
print("find the position of the first occurrence of a value greater than a given value")
# Input:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
# Solution:
print(np.argwhere(iris[:, 3].astype(float) > 1.0)[0])
print('\n')

#get the positions of top n values from a numpy array
print("get the positions of top n values from a numpy array")
# Input
np.random.seed(100)
a = np.random.uniform(1,50, 20)
print(a)

# Solution:
print(a.argsort())  #prints the index after sorting in ascending order
#> [18 7 3 10 15]
# Solution 2:
print(np.argpartition(-a, 5)[:10])
#> [15 10  3  7 18]
# Below methods will get you the values.
# Method 1:
print(a[a.argsort()][-5:])

# Method 2:
print(np.sort(a)[-5:])

# Method 3:
print(np.partition(a, kth=-5)[-5:])

# Method 4:
print(a[np.argpartition(-a, 5)][:5])

print('\n')

#convert an array of arrays into a flat 1d array
print("convert an array of arrays into a flat 1d array")
 # Input:
arr1 = np.arange(3)
arr2 = np.arange(3,7)
arr3 = np.arange(7,10)
array_of_arrays = np.array([arr1, arr2, arr3])
print('array_of_arrays: ', array_of_arrays)
# Solution 1
arr_2d = np.array([a for arr in array_of_arrays for a in arr])
print(arr_2d)
# Solution 2:
arr_2d = np.concatenate(array_of_arrays)
print(arr_2d)
print('\n')


#find the duplicate records in a numpy array
print("Find the duplicate entries (2nd occurrence onwards) in the given numpy array and mark them as True. First time occurrences should be False")
# Input
a = np.random.randint(0, 5, 10)
print(a)
print(a.shape[0])
## Solution

# Create an all True array
out = np.full(a.shape[0], True)

# Find the index positions of unique elements
unique_positions = np.unique(a, return_index=True)[1]

# Mark those positions as False
out[unique_positions] = False

print(out)
print('\n')

#convert a PIL image to numpy array
print("convert a PIL image to numpy array")
from io import BytesIO
from PIL import Image
import PIL, requests
# Import image from URL
URL = 'https://upload.wikimedia.org/wikipedia/commons/8/8b/Denali_Mt_McKinley.jpg'
response = requests.get(URL)
# Read it as Image
I = Image.open(BytesIO(response.content))
# Optionally resize
I = I.resize([150,150])
# Convert to numpy array
arr = np.asarray(I)
print(arr)
# Optionaly Convert it back to an image and show
im = PIL.Image.fromarray(np.uint8(arr))
##Image.Image.show(im)
print('\n')


#drop all missing values from a numpy array
print("Drop all nan values from a 1D numpy array")
a = np.array([1,2,3,np.nan,5,6,7,np.nan])
b= a[~np.isnan(a)]
print(a)
print(b)
print('\n')


#compute the euclidean distance between two arrays
print("compute the euclidean distance between two arrays")
a = np.array([1,2,3,4,5])
b = np.array([4,5,6,7,8])
# Solution
dist = np.linalg.norm(a-b)
print(dist)

