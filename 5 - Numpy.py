import numpy as np

#to use numpy, you should place lists into arrays - either 1d or 2d arrays
#for example, the below is a 2d array
my_mat = [[1,2,3],[4,5,6],[7,8,9]]
print(np.array(my_mat))             #prints a 2d array

#creating lists using numpy
print(np.arange(0,10,2))        #prints only even numbers

#Creating a list of 10 evenly spaced numbers between a certain range - namely number 0 and 5
print(np.linspace(0,5,10))

#using andom numbers between 0 and 1
print(np.random.rand(5))
#printing random numbers based on the normal distribution curve
print(np.random.randn(5))

#printing random number of integers between two numbers
print(np.random.randint(1,100,10))          #prints 10 random integers between 1 and 100 (excl 100)


#finding the location of the random numbers generated
arr = np.arange(25)
print(arr)
ranarr = np.random.randint(1,100,25)
print(ranarr)
print(ranarr.argmax())                  #prints as 3 - meaning the max value is located at position 4 in the array
print(ranarr.argmin())

print(ranarr.dtype)                     #print the type of object it is

######################################################################################################################

arr = np.arange(11)

#If you want to change the values in the array to be a certain other value permanently:
arr[0:5] = 99
print(arr)                              #Prints the first 5 numbers in the array to be 99

#If you want the values in the array to be changed temporarily to another constant value:
arr_copy = arr.copy()
arr_copy[:] = 100
print(arr_copy)                         #Print all the values to be temporarily 100
print(arr)                              #However, when you print the original array, it is still with thee 5 99 values

#How to slice matrices in a 2d array (matrix)
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
print(arr_2d)
print("\tGrabbing elements from a 2d array:\n")
print(arr_2d[1,2])                      #Grabs the number 30
print(arr_2d[:2,1:3])                   #Grabs the first 2 rows and the second and third columns


#Create a 5 row and 10 column matrix and grab elements within it
new_2d = np.arange(50).reshape(5,10)
print(new_2d)

#############################Exercises####################################################
print('#################################################')

#Create an array of 10 zeros
print(np.zeros(10))

#Create an array of 10 ones
print(np.ones(10))

#Create an array of 10 5's
print(np.ones(10) * 5)

print((np.arange(0,9).reshape(3,3)))

print(np.random.rand(1))        #Prints random number between 0 and 1

print(np.linspace(0,1,100).reshape(10,10))


mat = np.arange(1,26).reshape(5,5)
print(mat.sum())                #Prints sum of all the numbers in the array
print(mat.std())
