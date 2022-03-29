from lib.plot_utils import plot_basis, plot_vector
from numpy import meshgrid
from lib.utils import get_rotation_matrix_to_rotate_vector_a_to_vector_b

def algorithm(t,omega,acc,fig,ax):
    ax.clear()
    R = get_rotation_matrix_to_rotate_vector_a_to_vector_b(a=acc,b=[0,0,-1])
    g = R @ acc
    plot_vector(ax=ax,v=acc)
    plot_vector(ax=ax,v=g,color=(1,0,0))
    ax.legend(['acc','g'])
    # plot_basis(ax,R=R)

def template(t,omega,acc,fig,ax):
    ax.clear()
    R = get_rotation_matrix_to_rotate_vector_a_to_vector_b(a=[0,0,-1],b=acc)
    plot_vector(ax=ax,v=acc)
    plot_basis(ax,R=R)
def algorithm3(t,omega,acc,fig,ax):
    print(acc)
    xx, yy = meshgrid([-.25,.25],[-.5,.5])
    z = (-acc[0] * xx - acc[1] * yy - 0) * 1. /acc[2]
    ax.clear()
    ax.plot_surface(xx, yy, z)
    plot_vector(ax=ax,v=acc)
