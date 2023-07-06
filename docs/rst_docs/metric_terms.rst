.. container:: flushleft

   |  **Metric Terms in Pyrokinetics**
   | **H.G. Dudding**
   |  Email: harry.dudding@ukaea.uk

.. _`sec:TorCoords`:

Toroidal coordinate system
==========================

.. _toroidal coord transforms:

Metric terms
------------

The transformations from cylindrical coordinates
:math:`\left \{R, \Phi, Z \right \}` to toroidal coordinates
:math:`\left \{r, \theta, \zeta \right \}` are

.. math::

   \begin{aligned}
   r&=r \left(R, Z \right)   &  R &=R \left(r, \theta \right)   \\
   \theta&=\theta\left(R, Z \right) &  \Phi&=- \zeta \\
   \zeta&= - \Phi   &  Z&=Z\left(r, \theta \right)
   \end{aligned}

where the exact relations are defined through the flux-surface
parameterisation chosen. Here :math:`\Phi` is oriented anti-clockwise
when viewed from above, :math:`\zeta` is oriented clockwise, and
:math:`\theta` is oriented anti-clockwise around the flux surface, such
that :math:`\{R, \Phi, Z \}` and :math:`\{r, \theta, \zeta \}` form
right-handed coordinate systems.

The covariant metric components
(:math:`g_{i j} = \partial \mathbf{r} / \partial \xi^i \cdot \partial \mathbf{r} / \partial \xi^ j`,
for position vector :math:`\mathbf{r}` and general coordinates
:math:`\xi^i`) are

.. math:: g_{r r} = \left(\frac{\partial R}{\partial r}\right)^2 + \left(\frac{\partial Z}{\partial r }\right)^2

.. math:: g_{r \theta} = \frac{\partial R}{\partial r}\frac{\partial R}{\partial \theta} + \frac{\partial Z}{\partial r}\frac{\partial Z}{\partial \theta}

.. math:: g_{\theta \theta} = \left(\frac{\partial R}{\partial \theta}\right)^2 + \left(\frac{\partial Z}{\partial \theta}\right)^2

.. math:: g_{r \zeta} = g_{\theta \zeta} = 0

.. math:: g_{\zeta \zeta} = R^2

with the Jacobian

.. math::

   \mathcal{J}_{r \theta \zeta} = R\left(\frac{\partial R}{\partial r} \frac{\partial Z}{\partial \theta} - \frac{\partial R}{\partial \theta}\frac{\partial Z}{\partial r} \right).
   \label{eq:toroidal Jacob}

Note that in general :math:`g_{r \theta} \neq 0` and thus the toroidal
coordinate system is non-orthogonal.

The contravariant metric components
(:math:`g^{i j} = \nabla \xi^i \cdot \nabla \xi^j`) are

.. math:: g^{r r} = \frac{g_{\theta \theta} g_{\zeta \zeta}}{\left(\mathcal{J}_{r \theta \zeta}\right)^2}

.. math:: g^{r \theta} = - \frac{g_{r \theta} g_{\zeta \zeta }}{\left(\mathcal{J}_{r \theta \zeta}\right)^2}

.. math:: g^{\theta \theta} = \frac{g_{r r} g_{\zeta \zeta}}{\left(\mathcal{J}_{r \theta \zeta}\right)^2}

.. math:: g^{r \zeta} = g^{\theta \zeta} = 0

.. math:: g^{\zeta \zeta} = \frac{1}{g_{\zeta \zeta}}.

Physical quantities
-------------------

The magnetic field can be written

.. math::

   \begin{split}
           \mathbf{B} & = \psi^{\prime} \nabla r \times \nabla \left[ q\left(r\right) \theta - \zeta + G_0\left(r, \theta\right) \right] \\
           & =  \psi' \nabla \zeta \times \nabla r + B_{\zeta}\left(r\right) \nabla \zeta
   \end{split}

where :math:`\psi^{\prime} = \mathrm{d} \psi / \mathrm{d} r` is the
radial derivative of the poloidal magnetic flux divided by
:math:`2 \pi`, :math:`q(r)` is the safety factor, :math:`B_{\zeta}(r)`
is the current function

.. math::

   B_{\zeta}\left( r \right) = \frac{q \psi^{\prime}}{\left \langle \mathcal{J}_{r \theta \zeta} g^{\zeta \zeta} \right \rangle}
       \label{eq: B zeta def}

where
:math:`\left \langle f \right \rangle = \frac{1}{2 \pi} \int_{- \pi}^{\pi} f \, \mathrm{d} \theta`
denotes a poloidal average and :math:`G_0(r, \theta)` is a periodic
function in :math:`\theta` which satisfies

.. math::

   \frac{\partial G_0}{\partial \theta} = \frac{B_{\zeta}}{\psi^{\prime}} \mathcal{J}_{r \theta \zeta} g^{\zeta \zeta} - q.
   \label{eq:G0 deriv}

The current density is

.. math:: \mu_0 \mathbf{J} = \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r} \nabla r \times \nabla \zeta - \frac{1}{\psi^{\prime}} \left(B_{\zeta}  \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r}  + g_{\zeta \zeta} \mu_0 \frac{\mathrm{d}P}{\mathrm{d} r}  \right) \nabla \zeta

where, from the local Grad-Shafranov solution, we have

.. math::

   \begin{split}
       \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r} = \frac{B_{\zeta}}{H} \Bigg(\frac{q'}{q} \left \langle \mathcal{J}_{r \theta \zeta} g^{\zeta \zeta} \right \rangle & - \left \langle \mathcal{J}_{r \theta \zeta} \frac{\partial g^{\zeta \zeta}}{\partial r} \right \rangle - \left[ \frac{\mu_0}{\left( \psi^{\prime} \right)^2} \frac{\mathrm{d} P}{\mathrm{d} r} \right] \left \langle \frac{\left(\mathcal{J}_{r \theta \zeta}\right)^3 g^{\zeta \zeta}}{g_{\theta \theta}} \right \rangle \\ &  + \left \langle \frac{\mathcal{J}_{r \theta \zeta} g^{\zeta \zeta}}{g_{\theta \theta}} \left( \frac{\partial g_{r \theta}}{\partial \theta} - \frac{\partial g_{\theta \theta}}{\partial r} - \frac{g_{r \theta}}{\mathcal{J}_{r \theta \zeta}} \frac{\partial \mathcal{J}_{r \theta \zeta}}{\partial \theta} \right) \right \rangle \Bigg )
   \end{split}
   \label{eq:dBzetadr}

where :math:`P = P\left( r \right)` is the equilibrium plasma pressure,
:math:`q^{\prime} = \mathrm{d}q/\mathrm{d} r` is the radial derivative
of the safety factor and

.. math::

   H = \left \langle \mathcal{J}_{r \theta \zeta} g^{\zeta \zeta} \right \rangle + \left(\frac{B_{\zeta}}{\psi^{\prime}}\right)^2 \left \langle \frac{\left( \mathcal{J}_{r \theta \zeta}\right)^3 \left(g^{\zeta \zeta}\right)^2}{g_{\theta \theta} } \right \rangle.
   \label{eq:H}

The Grad-Shafranov solution also constrains the radial derivative of the
Jacobian to be

.. math::

   \begin{split}
       \frac{\partial \mathcal{J}_{r \theta \zeta}}{\partial r} = \, & \mathcal{J}_{r \theta \zeta} \frac{\psi''}{\psi'} - \frac{\mathcal{J}_{r \theta \zeta}}{g_{\theta \theta}} \left( \frac{\partial g_{r \theta}}{\partial \theta} - \frac{\partial g_{\theta \theta}}{\partial r} - \frac{g_{r \theta}}{\mathcal{J}_{r \theta \zeta}} \frac{\partial \mathcal{J}_{r \theta \zeta}}{\partial \theta} \right) \\ & + \frac{\left(\mathcal{J}_{r \theta \zeta}\right)^3}{g_{\theta \theta}} \left[\frac{\mu_0}{\left( \psi^{\prime} \right)^2} \frac{\mathrm{d} P}{\mathrm{d} r} \right] + \frac{\left(\mathcal{J}_{r \theta \zeta}\right)^3 g^{\zeta \zeta}}{g_{\theta \theta}} \left[ \frac{B_{\zeta}}{\left(\psi' \right)^2} \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r} \right].
   \end{split}
   \label{eq:Radial_Deriv_Jacobian}

where :math:`\psi^{\prime \prime} = \mathrm{d}^2 \psi / \mathrm{d} r^2`
is arbitrary, due to it always appearing as part of the combination
:math:`\partial \mathcal{J}_{r \theta \zeta} / \partial r - \mathcal{J}_{r \theta \zeta} \left( \psi'' / \psi' \right)`
in quantities of interest for local equilibria.

.. _`sec:global field aligned`:

Global field-aligned coordinates
================================

Metric terms
------------

To obtain field-aligned coordinates we transform from the toroidal
system to the global field-aligned system,
:math:`\{r, \theta, \zeta \} \rightarrow \{r, \alpha, \theta \}`,
defined by

.. math::

   \begin{aligned}
   r&=r   &  
   r &= r  \\
   \alpha &= \sigma_{\alpha} \left[q\left( r \right) \theta - \zeta + G_0 \left(r, \theta \right) \right] & 
   \theta & = \theta \\
   \theta &=\theta   & 
   \zeta &=q\left( r \right) \theta - \sigma_{\alpha} \alpha + G_0 \left(r, \theta \right)
   \end{aligned}

where :math:`\sigma_{\alpha}` takes values of either :math:`1` or
:math:`-1`. For :math:`\sigma_{\alpha} = 1`,
:math:`\{r, \alpha, \theta \}` forms a right-handed system, such as in
GENE, whereas with :math:`\sigma_{\alpha} = -1` then
:math:`\{r, \theta, \alpha\}` forms a right-handed system, such as in
CGYRO. The covariant metric components are, using
:math:`\partial \zeta / \partial r = \sigma_{\alpha} \partial \alpha / \partial r`,
:math:`\partial \zeta / \partial \theta = \sigma_{\alpha} \partial \alpha / \partial \theta`
and using a tilde to denote the field-aligned system,

.. math:: \tilde{g}_{rr} = g_{r r} + \left(\frac{\partial \alpha}{\partial r} \right)^2 g_{\zeta \zeta}

.. math:: \tilde{g}_{r \alpha} = - \frac{\partial \alpha}{\partial r} g_{\zeta \zeta}

.. math:: \tilde{g}_{r \theta} = g_{r \theta } + \frac{\partial \alpha}{\partial r} \frac{\partial \alpha}{\partial \theta} g_{\zeta \zeta}

.. math:: \tilde{g}_{\alpha \alpha} = g_{\zeta \zeta}

.. math:: \tilde{g}_{\alpha \theta} = - \frac{\partial \alpha}{\partial \theta} g_{\zeta \zeta}

.. math:: \tilde{g}_{\theta \theta } = g_{\theta \theta  } + \left( \frac{\partial \alpha}{\partial \theta}\right)^2 g_{\zeta \zeta}.

where :math:`\partial \alpha / \partial r` and
:math:`\partial \alpha / \partial \theta` are calculated below. Note
that the Jacobian remains unchanged from the toroidal system. The
contravariant metric components are

.. math:: \tilde{g}^{r r} = g^{r r}

.. math::

   \tilde{g}^{r \alpha} = \frac{\partial \alpha}{\partial r} g^{r r} + \frac{\partial \alpha}{\partial \theta} g^{r \theta} 
   \label{eq:eqgralpha}

.. math::

   \tilde{g}^{\alpha \alpha} = \left( \frac{\partial \alpha}{\partial r} \right)^2 g^{r r} + 2 \frac{\partial \alpha}{\partial r} \frac{\partial \alpha}{\partial \theta} g^{r \theta} + \left( \frac{\partial \alpha}{\partial \theta} \right)^2 g^{\theta \theta} + g^{\zeta \zeta}.
   \label{eq:g^alphaalpha}

.. math:: \tilde{g}^{r \theta} = g^{r \theta}

.. math:: \tilde{g}^{\theta \theta} = g^{\theta \theta}

.. math:: \tilde{g}^{\alpha \theta} = \frac{\partial \alpha}{\partial r} g^{r \theta} + \frac{\partial \alpha}{\partial \theta} g^{\theta \theta}.

Evaluating :math:`\partial \alpha / \partial \theta`, we find

.. math::

   \begin{split}
           \frac{\partial \alpha}{\partial \theta} & = \sigma_{\alpha} \left[ q + \frac{\partial G_0}{\partial \theta} \right] \\
           & = \sigma_{\alpha} \left [\frac{B_{\zeta}}{\psi^{\prime}} \mathcal{J}_{r \theta \zeta} g^{\zeta \zeta} \right].
   \end{split}
   \label{eq:dalphadtheta}

where we have used equation `[eq:G0 deriv] <#eq:G0 deriv>`__. To then
calculate :math:`\partial \alpha / \partial r`, we differentiate
equation `[eq:dalphadtheta] <#eq:dalphadtheta>`__ with respect to
:math:`r` and integrate over :math:`\theta`. First differentiating, we
get

.. math::

   \begin{split}
       \frac{\partial^2 \alpha}{\partial r \partial \theta} & = \sigma_{\alpha} \left[ \frac{\mathrm{d} q}{\mathrm{d} r} + \frac{\partial^2 G_0}{\partial r \partial \theta} \right] \\ & = \sigma_{\alpha} \left[ \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r} \frac{\mathcal{J}_{r \theta \zeta} g^{\zeta \zeta}}{\psi^{\prime}}  +  \frac{B_{\zeta}}{\psi^{\prime}}  g^{\zeta \zeta} \left(\frac{\partial \mathcal{J}_{r \theta \zeta}}{\partial r} - \frac{\psi^{\prime \prime}}{ \psi^{\prime} } \mathcal{J}_{r \theta \zeta} \right) +  \frac{B_{\zeta}}{\psi^{\prime}} \frac{\partial g^{\zeta \zeta} }{\partial r}\mathcal{J}_{r \theta \zeta} \right].
   \end{split}

where the forms of :math:`\mathrm{d}B_{\zeta} / \mathrm{d} r` and
:math:`\partial \mathcal{J}_{r \theta \zeta} / \partial r` are given by
equations `[eq:dBzetadr] <#eq:dBzetadr>`__ and
`[eq:Radial_Deriv_Jacobian] <#eq:Radial_Deriv_Jacobian>`__. Now
integrating over :math:`\theta`,

.. math::

   \begin{split}
       \frac{\partial \alpha}{\partial r}  = \sigma_{\alpha} \int_{0}^{\theta} \left[ \frac{\mathrm{d} B_{\zeta}}{\mathrm{d} r} \frac{\mathcal{J}_{r \theta \zeta} g^{\zeta \zeta}}{\psi^{\prime}}  +  \frac{B_{\zeta}}{\psi^{\prime}}  g^{\zeta \zeta} \left(\frac{\partial \mathcal{J}_{r \theta \zeta}}{\partial r} - \frac{\psi^{\prime \prime}}{ \psi^{\prime} } \mathcal{J}_{r \theta \zeta} \right) +  \frac{B_{\zeta}}{\psi^{\prime}} \frac{\partial g^{\zeta \zeta} }{\partial r}\mathcal{J}_{r \theta \zeta} \right] \, \mathrm{d} \theta^{\prime}
   \label{eq:dalphadr}
   \end{split}

where
:math:`\left. \partial \alpha / \partial r \right\vert_{\theta = 0} = 0`
is assumed. Note that :math:`\partial \alpha / \partial r` is not
periodic, but satisfies

.. math:: \left . \frac{\partial \alpha}{\partial r} \right\vert_{\theta + 2 M \pi} = \left . \frac{\partial \alpha}{\partial r} \right\vert_{\theta} + \sigma_{\alpha} \frac{\mathrm{d} q}{\mathrm{d} r} 2 M \pi.
