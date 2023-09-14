.. container:: flushleft

   | **A. Bokshi**
   | arka.bokshi@gmail.com

.. _`sec:FlowShear`:

Flow shear
==========

The radial shear in the perpendicular flow profile :math:`\mathbf{v}_E` has a stabilizing impact on the linearly unstable modes and is implemented in almost all codes. We are interested in evaluating

.. math::
  :label: gammaE

  \begin{split}
        \gamma_E &= \frac{\partial}{\partial r} \left( \mathbf{v}_E. \frac{\mathbf{B}\times \nabla \psi}{\left|\mathbf{B}\times \nabla \psi \right|}  \right) \\
                &= \frac{\partial}{\partial r} \left( \frac{RB_p}{B} \frac{\partial \phi}{\partial \psi} \right)
  \end{split}

where we have used
  
.. math::
 :label: B

   \begin{split}
        \mathbf{B} = f(\psi) \nabla \zeta + \nabla \zeta \times \nabla \psi 
  \end{split}

.. math::
 :label: velE

   \begin{split}
        \mathbf{v}_E = \frac{\mathbf{B} \times \nabla \phi}{B} 
  \end{split}
and :math:`f(\psi)=RB_\zeta`. The electrostatic potential :math:`\phi` is determined from the radial force balance. For a species :math:`a`, the fluid equation of motion in steady-state reads

.. math::
  :label: fluid-equation

  \begin{split}
        0 = q_a n_a (\mathbf{E} + \mathbf{v}_0 \times \mathbf{B}) - \nabla p_a
  \end{split}

The equilibrium flow velocity is predominantly toroidal, i.e. :math:`\mathbf{v}_0=R \Omega_0 \nabla \zeta/|\nabla \zeta|`, yielding

.. math::
  :label: phi

  \begin{equation}
       \nabla \phi = -\frac{1}{q_a n_a} \nabla p_a  - \Omega_0 \nabla \psi
  \end{equation}

Some straightforward algebra yields the expression for shearing rate:

.. math::

  \begin{split}
       \gamma_E &= -\frac{RB_p}{B} \frac{\partial}{\partial r} \left[ \frac{1}{q_a n_a} \frac{\partial p_a}{\partial \psi}  + \Omega_0 \right] \\
                &= \frac{1}{B} \left(\frac{\partial \psi}{\partial r} \right)^2\left[ \frac{1}{(1+\eta_a)p_a n_a q_a } \left( \frac{\partial p_a}{\partial \psi}\right)^2  - \frac{1}{n_a q_a}\frac{\partial^2 p_a}{\partial \psi^2} \right] - \frac{r}{q} \frac{\partial \Omega_0}{\partial r}\frac{B_\zeta}{B}
  \end{split}

Note that :math:`\eta_a = (n_a T_a')/(T_a n_a')` (with prime denoting derivative with respect to :math:`\psi`) and :math:`\partial \psi/\partial r = RB_p`. `Giacomin et al <https://arxiv.org/pdf/2307.01669.pdf>`_. have found that the diamagnetic contribution is essential to obtaining saturated turbulence simulations.

