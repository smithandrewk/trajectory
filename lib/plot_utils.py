import matplotlib.pyplot as plt
from numpy import array
from numpy import eye,zeros
def make_3d_axes(fig=None,lim=1):
    if(fig==None):
        fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig,ax
def plot_vector(ax,v=[0,0,-1],lim=1,origin=array([0,0,0]),color=(0,.5,0)):
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    ax.quiver(*origin,*v,color=color)
def plot_basis(ax,R=eye(3),lim=1,linestyle='-'):
    # RBG
    RED = (.6,0,0)
    GREEN = (0,.6,0)
    BLUE = (0,0,.6)
    """
    let [color_1, color_2, ..., color_n] be the list you obtained after step 2, you should specify colors=[color_1, color_2, ..., color_n, color_1, color_1, color_2, color_2, ..., color_n, color_n]since actually the "-" part (consisting of 1 line) of all the non-zero arrows "->" wil be drawn first, then comes the ">" part (consisting of 2 lines).
    """
    basis_colors = [RED,BLUE,GREEN,RED,RED,BLUE,BLUE,GREEN,GREEN]
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))
    ax.quiver(*zeros((3,3)),*R,colors=basis_colors,label=['x'],linestyle=linestyle)
    ax.legend()