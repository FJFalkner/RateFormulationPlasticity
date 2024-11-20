import numpy as np
import matplotlib.pyplot as plt



# yield function
def vonMises(sig):
    f = np.sqrt(sig[0]**2 - sig[0]*sig[1] + sig[1]**2) - 235
    return f

# flow rule
def flowRule(sig):
    n = 0.5/np.sqrt(sig[0]**2 - sig[0]*sig[1] + sig[1]**2)*np.array([2*sig[0] - sig[1],2*sig[1] - sig[0]])
    return n


# material properties
E = 210000
nu = 0.3
Rp = 235
C = E/(1-nu**2)*np.array([[1, nu],[nu, 1]])

sigR = np.zeros([2,1])
sigR[0] = 235
sigR[1] = 0

m = 5
sigM = np.zeros([2, m+1])

# final strain
strain = np.zeros([2, 1])
strain[0] = 5E-3
strain[1] = 0E-3

# elastic limit
sig = C @ strain
scal = Rp/np.sqrt(sig[0]**2 - sig[0]*sig[1] + sig[1]**2)
sigM[:,1] = scal*sig.T

# strain to final strain (in m steps)
strInc = np.zeros([2, m+1])
strInc[0,1:m+1] = np.linspace(scal*strain[0],strain[0],m).T
strInc[1,1:m+1] = np.linspace(scal*strain[1],strain[1],m).T

# plastic strain 
strPlM = np.zeros([2, m+1])

h = 1E-8
for j in range(1,m):

    n = flowRule(sigM[:,j])

    dLambda = 0
    # determine plastic strain via Newton method
    for i in range(10):
        # stress and yield function
        sig0 = C @ (strInc[:,j+1] - strPlM[:,j] - dLambda*n)
        f0 = vonMises(sig0)

        # check convergence
        if f0 < 1E-10:
            break

        # numerical tangent
        # pertubated stress
        sigP = C @ (strInc[:,j+1] - strPlM[:,j] - (dLambda + h)*n)
        fP = vonMises(sigP)
        # tangent
        dfdLambda = (fP - f0)/h

        # update dLambda
        dLambda += -f0/dfdLambda

    # plastic strain increment
    strPl = dLambda*n.T
    # stress
    sigM[:,j+1] = C @ (strInc[:,j+1] - strPlM[:,j] - strPl).T
    # plastic strain
    strPlM[:,j+1] = strPlM[:,j] + strPl

# plot stresses
plt.plot(sigM[0, :], sigM[1, :], marker='o')

# Setting equal scaling for both axes
plt.axis('equal')

# Adding grid
plt.grid(True)

# Adding labels
plt.xlabel('sig_1')
plt.ylabel('sig_2')

# plot von Mises for plane stress:

# Define the implicit function for plotting
def implicit_function(x, y):
    return np.sqrt(x**2 - x*y + y**2) - 235

# Create a grid of points
x = np.linspace(-300, 300, 400)
y = np.linspace(-300, 300, 400)
X, Y = np.meshgrid(x, y)
Z = implicit_function(X, Y)

# Plot the implicit function
plt.contour(X, Y, Z, levels=[0], colors='r')

# Show the plot with the implicit function
plt.show()