# Gyroscopic Pipeline
We want to determine the *orientation* of a rigid body in 3 dimensions based on its *Angular Velocity*. 

![fig](cube_rotation.gif)

In this pipeline, we consider the rigid body's *position* to be fixed. In the future, we will consider position not to be fixed; however, we make this idealization to consider gyration first.

**Before proceeding with a 3 dimensional mathematical process, let us consider an idealization in 2 dimensions.**

## 2 Dimensional Idealization
Imagine you live in _flatland_, 

![fig](fig.svg)

a 2-dimensional land whererin you are a 1-dimensional creature (with a rigid-ish body).

![you](you.svg)

You can *rotate* 20 degrees about the *origin*, labeled 'O', resulting in an *orientation* of 20 degrees counterclockwise about the origin.

![you_20_degrees](you_rot_20.svg)

You can also rotate 90 degrees, or pi/2 radians.

![you_90_degrees](you_rot_90.svg)

So, we can define your (or any rigid body's) orientation by the angle rotated minus the initial angle. 

## Formalism
Let {{< katex >}}A{{< /katex >}} be a rigid body centered at the origin in a 2 dimensional cartesian space. Define the initial orientation of the rigid body by the basis 
{{< katex display >}}
B
=
\begin{bmatrix}
   i & j \\
\end{bmatrix}
=
\begin{bmatrix}
   1 & 0 \\
   0 & 1
\end{bmatrix}
{{< /katex >}}
shown in the following figure where *i* is red and *j* is green.

![](initial.svg)

The resulting *orientation* after a rotation of {{< katex >}}\theta{{< /katex >}} radians can be described by the following rotation matrix:

*see page on rotation matrices*
{{< katex display >}}
R
=
\begin{bmatrix}
   \cos(\theta) & -\sin(\theta) \\
   \sin(\theta) & \cos(\theta)
\end{bmatrix}.
{{< /katex >}}

We obtain the new basis by 

{{< katex display >}}
RB
=
\begin{bmatrix}
   \cos(\theta) & -\sin(\theta) \\
   \sin(\theta) & \cos(\theta)
\end{bmatrix}
\begin{bmatrix}
   1 & 0 \\
   0 & 1
\end{bmatrix}=R.
{{< /katex >}}

## Example
Suppose 
{{< katex >}}\theta=\frac{\pi}{2}{{< /katex >}}
radians. Then we have the following orientation:
{{< katex display >}}
RB
=
\begin{bmatrix}
   \cos(\frac{\pi}{2}) & -\sin(\frac{\pi}{2}) \\
   \sin(\frac{\pi}{2}) & \cos(\frac{\pi}{2})
\end{bmatrix}
\begin{bmatrix}
   1 & 0 \\
   0 & 1
\end{bmatrix}
{{< /katex >}}

{{< katex display>}}
=
\begin{bmatrix}
   \cos(\frac{\pi}{2}) & -\sin(\frac{\pi}{2}) \\
   \sin(\frac{\pi}{2}) & \cos(\frac{\pi}{2})
\end{bmatrix}
{{< /katex >}}

{{< katex display>}}
=
\begin{bmatrix}
   0 & -1 \\
   1 & 0
\end{bmatrix}
{{< /katex >}}


{{< katex display>}}
=
\begin{bmatrix}
   i & j \\
\end{bmatrix}.
{{< /katex >}}

{{<columns>}}

![gif](final.svg)

![gif](out.gif)

# Gyroscope
Suppose the rigid body has a gyroscope which is located on the rigid body at the origin. As the rigid body rotates, so does the gyroscope. The gyroscope has a sampling rate f, and the gyroscope provides a 'reading' every 1/f seconds. Suppose this gyroscope has 1 degree of freedom (natural for a gyroscope living in a 2 dimensional world); thus, a reading from the gyroscope consists of 1 value, angular velocity, omega. Suppose this gyroscope also outputs a timestamp t with omega.

Suppose the rigid body is at rest, then the reading from the gyroscope will be (omega,t) = (0 rad/s, t)