#A kernel is just an instance of Python running

#Using the modulus symbol "%" returns the decimals remaining after you divide 2 numbers
print(25 % 6)

num = 2
name = "Veliko"
print("My number is {} and my name is {}".format(num, name))

#Obtaining elements from nested lists
my_list = [1, 2, [3, 4]]
print(my_list[2][1])            #Prints out the number 4


#What a set is - it is a number of unique items
t = {1,1,1,2,2,3,3,3,3,3}               #Prints only the unique numbers
print(t)


i = 1
while i < 5:
    print(f"i is: {i}")
    i = i + 1

#Difference between return in a function and print - return actually requires you to store the value in a variable.


#Functions, map() function and lamda
seq = [1,2,3,4,5]
output = []
for i in seq:
    value = i * 2
    output.append(value)
print(output)

#Instead of using a for loop, can use the map function
def times2(var):
    return var*2
print(times2(5))

#now using the map() function as a shortcut
print(list(map(times2, seq)))

#Using lamda - basically avoids you having to create a function all the time
n = lambda var:var * 2
print(n(6))            #This will print 12 - namely 6 x 2

#Using the split() function to split a sentence or string based on all its white spaces into a list
s = "My name is Veliko"
print(s.split())            #Splits each word and puts it as a element in a list


####################Exercises########################################################
def domainGet(email_site):
    return email_site.split('@')[-1]

print(domainGet('user@domain.com'))         #only prints the domain name


#print only words beeginning with the letter s
seq = ['soup','dog','salad','cat','great']
print(list(filter(lambda var: var[0] == 's', seq)))

