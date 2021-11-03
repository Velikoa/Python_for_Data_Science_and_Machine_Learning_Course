import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,11)
y = x ** 2

#There are 2 ways to graph data - using the functional or the object orientated way
#this is the Functional way
plt.plot(x,y)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title')

plt.show()

#Plotting more than one type of graph on same surface
plt.subplot(1,2,1)         #number of rows, number of columns, plot number referring to
plt.plot(x,y,'r')

plt.subplot(1,2,2)
plt.plot(y,x,'b')

plt.show()

#Using the object orientated method
fig = plt.figure()                  #Why doing this? This is basically creating the empty canvas to work on!
axes = fig.add_axes([0.1,0.1,0.8,0.8])

axes.plot(x,y)
axes.set_xlabel('X Label')
axes.set_ylabel('Y Label')
axes.set_title('Set Title')

plt.show()

fig_1 = plt.figure()
axes1 = fig_1.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig_1.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y)
axes1.set_title("LARGER PLOT")
axes2.plot(y,x)
axes2.set_title("SMALLER PLOT")

plt.show()

########################Next Video########################

fig_2, axes = plt.subplots(nrows=1,ncols=2)        #can specify the number of rows and columns to have in your canvas. Uses tuple unpacking.

axes[0].plot(x,y)
axes[0].set_title('First Plot')

axes[1].plot(y,x)
axes[1].set_title('Second Plot')

plt.tight_layout()              #Creates more space around the seperate plots

plt.show()



fig_3, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,2))       #Numbers are in inches - width (8 inches wide) and height

axes[0].plot(x,y)
axes[1].plot(y,x)

plt.tight_layout()

plt.show()

#Saving a figure to a file
#fig.savefig('mypicture.png')


fig_4 = plt.figure()
ax = fig_4.add_axes([0,0,1,1])
ax.plot(x,x**2, label='X Squared')
ax.plot(x,x**3, label='X Cubed')

ax.legend(loc=0)                 #specify where to have the legend. 0 = best location according to matplotlib

ax.set_title('Title')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()


fig_5 = plt.figure()
ax1 = fig_5.add_axes([0,0,1,1])

#Alpha shows how transparent the line plotted is. Linestyle can be dash or dotted line.
#Marker is the type of shape you want to be shown for eaech plotted point
ax1.plot(x,y,color='green', linewidth=3, alpha=0.5, linestyle='--', marker='o', markersize=10,
         markerfacecolor='yellow', markeredgewidth=3, markeredgecolor='purple')         #Change colour of the line plotted or can use RGB colours.


plt.show()



