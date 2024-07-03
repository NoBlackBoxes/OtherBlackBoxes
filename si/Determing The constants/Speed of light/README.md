The speed of light is used to derive the value of a metre so a logical place to begin determining a meter is a value for the speed of light.
To do this we will require the four Maxwell equations in a vacuum:

$\nabla\cdot E = 0$
$\nabla\cdot B = 0$
$\nabla\ E = - \frac{\partial B}{\partial t}$
$\nabla\times B = \mu_0\epsilon_0 \frac{\partial E}{\partial t}$

and the vector identity:
$\nabla\times V=\nabla\cdot\left(\nabla V\right)-\nabla^2 V$

We thus substitute in Maxwell 3 and Maxwell 4 into the vector identity to see what results out.

Starting with Maxwell 3:
\
$\nabla\times(\nabla\times E) =$
\
$\nabla\times (- \frac{\partial B}{\partial t}) =$
\
substituting in the right hand side of Maxwell 3 resulting in:
\
$\nabla\times(\nabla\times E) = \nabla\times (- \frac{\partial B}{\partial t})$


then continuing on the left hand side we expand out using the vector identity for:
\
$\nabla\cdot (\nabla\times E) - \nabla^2 E = \nabla\times (- \frac{\partial B}{\partial t})$
\
noting that curl of div results in zero it simplifies down to:
\
$- \nabla^2 E = \nabla\times (- \frac{\partial B}{\partial t})$
\


if we then view the partial derivitve as an operator, we can then rearrange the partial term resulting in
\
$- \nabla^2 E = \frac{\partial}{\partial t} (\nabla\times B)$
\
then using Maxwell 4 we recieve the relation
\
$- \nabla^2 = \mu_0 \epsilon_0 \frac{\partial^2 E}{\partial t^2}$

we then repeat this for Maxwell 4 but notice instead the need to substitute Maxwell 3:

$\nabla\times (\nabla\times B)$
\
$= \nabla\times (\mu_0)\epsilon_0 \frac{\partial E}{\partial t}$

\
$\nabla\times (\nabla\times B) = \nabla\times (\mu_0) \epsilon_0 \frac{\partial E}{\partial t}$
\


LHS:
\
$\nabla\cdot (\nabla\times B) - \nabla^2 B = \nabla\times (\mu_0)\epsilon_0 \frac{\partial E}{\partial t}$
\
$- \nabla^2 B = \nabla\times (\mu_0) \epsilon_0 \frac{\partial E}{\partial t}$
\
$- \nabla^2 B = \frac{\partial}{\partial t}(\nabla\times E)$
\
$- \nabla^2 B =  \mu_0\epsilon_0\frac{\partial^2 B}{\partial t^2}$


these clearly both match the form of the wave equation ($\nabla^2 "wave" = \frac{1}{"speed"^2} \frac{\partial^2 "wave"}{\partial t^2}$), 
so we assume as they share constant terms that the speed of the electromagnetic wave (this idea of a single field comes from how closely related the two fields -
are and that thus they must be a the same field) or a wave of light, must be constatn at a value of

$c = \frac{1}{\sqrt(\mu_0\epsilon_0)}$

with this knowledge we must set about determining values of both $\mu_0$ and $\epsilon_0$ to calcualte c, to determine the length of a metre.



