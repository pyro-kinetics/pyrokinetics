.. container:: flushleft

   | **A. Bokshi**
   | arka.bokshi@gmail.com

.. _`sec:CollisionFrequency`:

Collision frequency
===================

The frequency with which a species :math:`a`, characterized by its mass :math:`m_a` and temperature :math:`T_a`, undergoes collisions with a background species :math:`b` with number density :math:`n_b`, is given by (in SI units)

.. math::
 :label: nuab

   \begin{equation}
           \nu_{ab} = \frac{ \sqrt{2}\pi n_b Z_a^2 Z_b^2 e^4 \ln \Lambda^{a/b} }{ (4\pi \epsilon_0)^2 \sqrt{m_a} T_a^{3/2} } 
   \end{equation}

The collision frequency as defined in various codes is written in the Pyrokinetics `normalization <https://github.com/pyro-kinetics/pyrokinetics#note-on-units>`_ convention.

.. _cgyro:
CGYRO
-----
CGYRO `defines <https://gafusion.github.io/doc/cgyro/cgyro_list.html#cgyro-nu-ee>`_ the normalized electron-electron collision frequency :math:`\nu_{ee}` in Gaussian units -- without the factor :math:`(4\pi\epsilon_0)^2`, i.e.

.. math::
   \begin{split}
           \nu_{ee} & = \frac{ 4\pi n_e e^4 \ln \Lambda }{ \sqrt{m_e} (2T_e)^{3/2} } 
   \end{split}
   \label{eq:nu_cgyro}

All other collision rates are self-consistently determined from this :math:`\nu_{ee}a/c_s`, which is equal to `NU_EE` in the input deck.

.. _gene:
GENE
----
In GENE, the parameter `coll` is defined as 

.. math::
   \begin{split}
           \nu_{c} &= \pi \ln \Lambda e^4 n_\mathrm{ref} L_\mathrm{ref} / (2^{3/2}T^2_\mathrm{ref}) \\
                        &= 2.3031\cdot 10^{-5} \frac{ L_\mathrm{ref} n^{19}_\mathrm{ref} } {(T^k_\mathrm{ref})^2} \ln \Lambda
   \end{split}
   \label{eq:nu_c}
(note the division by :math:`(4\pi \epsilon_0)^2`) and where

.. math::
   \begin{split}
           \ln \Lambda = 24 - \ln \left( \frac{\sqrt{n_e 10^{-6}}}{T_e}  \right)
   \end{split}

Working with the assumptions that in the calculation of `coll` we have used

.. math::
   \begin{split}
        n_\mathrm{ref}&=n_e\\
        T_\mathrm{ref}&=T_e\\
        m_\mathrm{ref}&=m_D
   \end{split}
i.e. :math:`c_s=\sqrt{T_e/m_D}`, \we arrive at

.. math::
   \begin{split}
         \hat{\nu}_{ee} &= \nu_{ee} \frac{a}{c_s} = 4 \sqrt{\frac{m_D}{m_e}} \nu_c \frac{a}{L_\mathrm{ref}}
   \end{split}

.. _gs2:
GS2
---
GS2 `defines <https://gyrokinetics.gitlab.io/gs2/page/namelists/index.html>`_ the normalized collision frequency `vnewk = nu_ss Lref/vref`, where from equation :eq:`nuab` it naturally follows that 

.. math::
   \begin{split}
           \nu_{ss} & = \frac{ \sqrt{2}\pi n_s Z_s^4 e^4 \ln \Lambda }{ (4\pi \epsilon_0)^2 \sqrt{m_s} T_s^{3/2} } 
   \end{split}
   \label{eq:nu_gs2}

Expressed in terms of the electron-electron collisions:

.. math::
   \begin{split}
           \nu_{ss} & = \frac{ \hat{n}_s Z_s^4 }{ \sqrt{\hat{m}_s} \hat{T}_s^{3/2} } \nu_{ee}
   \end{split}
   \label{eq:nu_ee}
where we have assumed that the Coulomb logarithm is roughly invariant and hat refers to normalization by the corresponding electron parameters. 

.. _gkw:
GKW
---
GKW `defines <https://bitbucket.org/gkw/gkw/src/develop/doc/manual/>`_ the normalized collision frequency :math:`\Gamma_N^{a/b} = \nu_{ab} R_\mathrm{ref}/v_\mathrm{tha}`, which yields 

.. math::
   \begin{split}
           \Gamma_{N}^{a/b} & = 6.5141.10^{-5} \frac{R_\mathrm{ref} n^{19}_b}{(T^k_a)^2} Z_a^2 Z_b^2 \ln \Lambda^{a/b}
   \end{split}
GKW can define all other collision frequencies in terms of the *singly charged* ion-ion collision frequency taken at some reference values, :math:`\Gamma_N^{i/i}` or `coll_freq`, by setting the switch `freq_override=.true.`

.. math::
   \begin{split}
           \Gamma_{N}^{i/i} = 6.5141 \cdot 10^{-5} \frac{R_\mathrm{ref} n^{19}_\mathrm{ref}}{(T^k_\mathrm{ref})^2} \ln \Lambda^{i/i}
   \end{split}
Here :math:`R_\mathrm{ref},n^{19}_\mathrm{ref},T^k_\mathrm{ref}` are in :math:`\mathrm{m},10^{19}/\mathrm{m}^3,\mathrm{keV}` units respectively. In the Pyrokinetics convention, beginning with :eq:`nuab`, and noting that :math:`\Gamma_N^{i/i} = \nu_{ii} R_\mathrm{ref}/v_\mathrm{thi}`, where :math:`v_\mathrm{thi}=\sqrt{2T_i/m_i}`, we derive

.. math::
   \begin{split}
           \hat{\nu}_{ee} = \nu_{ee} \frac{a}{c_s} = \sqrt{2} \frac{a}{R_\mathrm{ref}} \left( \frac{T_i}{T_e} \right)^2 \frac{n_e}{n_i} \sqrt{\frac{m_D}{m_e}} \Gamma^{i/i}_N
   \end{split}


.. _tglf:
TGLF
----
In TGLF, we define the `electron-ion collision frequency <https://gafusion.github.io/doc/tglf/tglf_list.html#xnue>`_, `XNUE`, which equals :math:`\nu_{ei} a/c_s`

.. math::
   \begin{split}
           \nu_{ei} = \frac{n_i}{n_e}\nu_{ee} 
   \end{split}

