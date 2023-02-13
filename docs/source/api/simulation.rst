Simulation
==========

Module that implement the simulation core functionalities.

The objective of the simulation is, for a set of assets 
:math:`A=(a_{1}, ..., a_{i}, ..., a_{m})`, knowing:

* Their price history :math:`P^{h}\in\large{R}^{N^{h}\times{m}}` for :math:`N^{h}` days.

* An initial price :math:`V\in\large{R}^{1\times{m}}` for each asset :math:`a_{i}`.

* The number of closing days :math:`N^{s}` where we have to simulate a price for 
  each asset.

* The expected return of :math:`A` at the end of the simulation 
  :math:`Er \in \large{R}^{1 \times{m} }`. We have 
  :math:`er_{i}=(p^{s}_{i,Nh} - p^{s}_{i,0})/p^{s}_{i,0}` the expected return of the 
  asset :math:`a_{i}` for the :math:`N^{s}_{f}` simulated days.

to simulate the future prices :math:`P^{s}\in\large{R}^{N^{s}\times{m}}` of \
:math:`A` such that:  

* :math:`C^{s}=C^{h}`

    :math:`C^{s}=Cov(Ed^{s})` \
    and :math:`C^{h}=Cov(Ed^{h})` \
    where :math:`Ed^{s}\in\large{R}^{N^{s}-1\times{m}}` represents the matrix of \
    the daily returns of the simulated asset prices :math:`A` and \
    :math:`Ed^{h}\in\large{R}^{N^{h}-1\times{m}}` those of their historical prices. \
    The daily return :math:`ed_{i,j}` of the asset :math:`a_{i}` on day :math:`j>1` \
    is such that :math:`ed_{i,j}=(p_{i,j} - p_{i,j-1})/p_{si,j-1}`.

* :math:`Er^{s}=Er`

    :math:`Er^{s} \in \large{R}^{1 \times{m} }` is the expected return of A over \
    the time period of the simulation.


1. Generate daily returns with a given covariance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To do this, we first generate a matrix :math:`M\in\large{R}^{N^{s}-1\times{m} }` \
composed of random values drawn from the standard normal distribution from which we \
will generate some simulated daily returns :math:`Q` such that :math:`Cov(Q)=C^{h}`.

The covariance matrix :math:`Cov(M)` is a symmetric positive definite real-valued \
matrix which is not exactly the identity matrix :math:`I`. We use the Choleski \
decomposition of :math:`Cov(M)` which tells us that there is a unique matrix \
:math:`L\in\large{R}^{m\times{m}}` upper triangular with real and positive diagonal \
entries such that :math:`Cov(M)=C^{M}=L^{T}L`.

We can do the same for :math:`C^{h}` and write that there is a unique upper triangular \
matrix :math:`L^{h}\in\large{R}^{m\times{m}}` such that :math:`C^{h}=L_{h}^{T}L_{h}`.

Now let's show that if the matrices :math:`L` and :math:`L^{h}` are invertible, then \
by setting :math:`Q=ML^{-1}L_{h}` it follows that :math:`Cov(Q)=C^{h}` .

To do this let's use the fact that :math:`\mathbb{E}(MA)=\mathbb{E}(M)A` by \
linearity of the expectation and the definition of covariance to write:

.. math:: 
    Cov(Q) &= \mathbb{E}[(MA - \mathbb{E}(MA))^{T}(MA - \mathbb{E}(MA)))]

    Cov(Q) &= \mathbb{E}[A^{T}(M - \mathbb{E}(M))^{T}(M - \mathbb{E}(M)))A]

    Cov(Q) &= A^{T}\mathbb{E}[(M - \mathbb{E}(M))(M - \mathbb{E}(M)))A]

    Cov(Q) &= A^{T}C^{M}A

    Cov(Q) &= L^{T}_{h}{L^{-1}}^{T}C^{M}L^{-1}L_{h}

or we have seen that by definition :math:`Cov(M)=C^{M}=L^{T}L` so we have \
:math:`{L^{-1}}^{T}C^{M}L^{-1}=I` and by the definition of :math:`L_{h}` we have:

.. math:: 
    Cov(Q) &= L^{T}_{h}L_{h}

    Cov(Q) &= C^{h}


2. Generate daily returns with the given expected return
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have a matrix of daily returns :math:`Q` that have the same covariance matrix :math:`C^{h}` as our historical daily price returns. But the vector of expected annual returns :math:`Er^{q}` that corresponds to these simulated daily returns is not the same as the original vector :math:`Er`.

By using the fact that, if :math:`c \in R^{N^{s}-1}` a constant vector then \
:math:`\mathbb{E}[Q+c^T]=\mathbb{E}[Q]`, we can demonstrate that we still have \
:math:`Cov(Q + c^T)=C^{h}`. (TODO Make the demo if needed).

So our goal is to found :math:`c` such that :math:`Q + c^T = Ed^{s}` .

Given an asset :math:`a_{u}`, with an expected return :math:`r_{u}` between days \
:math:`0` and :math:`t` we have:

.. math:: 1 + r_{u}=\prod_{i=1}^{t}(1 + r_{u,i})

If for all :math:`i` in :math:`[1,t]` we have :math:`r_{u,i} > -1` then:

.. math:: ln(1 + r_{u})=\sum_{k=1}^n{ln(1+r_{u,i})} 

.. note::
    The *Taylor* theorem :

    #. :math:`I` a subset of :math:`R`;
    #. :math:`a` in :math:`I`;
    #. :math:`E` a real normed vector space;
    #. :math:`f` a function of :math:`I` in :math:`E` derivable in :math:`a` up to a 
       certain order :math:`n\geq 1` .

    Then for any real number :math:`x` belonging to :math:`I` , we have the 
    *Taylor-Young* formula:

    .. math:: f(a+h)=\sum_{k=0}^n{\frac{f^{(k)}(a)}{k!}h^{k}+R_{n}(a+h)}

    where the remaining :math:`R_{n}(x)` is a negligible function with respect to \
    :math:`(x-a)^{n}` in the neighbourhood of :math:`a`.

If we apply the *Taylor theorem* to the logarithm function in 1 we have for all \
:math:`x > -1`:   

.. math:: ln(1+x)=\sum_{k=1}^{n} {(-1)^{k-1}\frac{x^{k}}{k}}+ R_{n}(1+x)

If :math:`x < 1` then the *Taylor-Young* formula stand that we have:

.. math:: R_{n}(1 + x)=o(x^{n})

In our particular case, we know that the daily returns :math:`r_{u, i}` are strictly \
less than 1 for all :math:`i` and :math:`u`. We can therefore always find a strictly \
positive integer :math:`n` such that :math:`ln(1+r_{u,i})` is approximated with a great \
accuracy by is corresponding polynomial *Taylor-Young* approximation of order :math:`n` \
. 

.. math::

    \lim_{n \to +\infty}ln(1 + r_{u,i}) - \sum_{k=1}^{n}{(-1)^{k-1}\frac{r_{u,i}^{k}}{k}} = 0


So for a given stock :math:`u` with  :math:`N^{s}_{f}` daily returns, and :math:`r_{u, i}` a \
    daily return at day :math:`i`, we can try to found the constant :math:`c_{u}` such \
    that for a simulated daily return :math:`q_{u,i}` we have \
    :math:`r_{u,i} = q_{u,i} + c_{u}` and:

.. math:: \sum_{i=1}^{N^{s}_{f}}{ln(1 + q_{u,i} + c_{u})} = ln(1 + r_{u})

To find :math:`c_{u}` we will solve an approximation of these equation by replacing \
    the logarithmic parts with their *Taylor-Young* approximation to create the \
    polynomial :math:`P_{u}^{n}(X)` of order :math:`n` such that:

.. math:: P_{u}^{n}(X) = \sum_{i=1}^{N^{s}_{f}} \sum_{k=1}^n(-1)^{k-1}{\frac{(q_{u,i} + X)^{k}}{k}} - ln(1 + r_{u})


Let be :math:`c_{n,u}` a root of :math:`P_{u}^{n}(X)`.  

If :math:`P_{u}^{n}(X)` as at least one real root :math:`c_{n,u}` such that \
:math:`| \underset{1 \leq i \leq n}{max}(q_{u,i}) + c_{n,u} | < 1` we take the  \
smallest one of these roots :math:`c_{n,u}` and we can observe that:  (TODO demo ?)

.. math:: \lim_{n \to +\infty}(c_{n,u}) = c_{u}


Our algorithm fails if :math:`P_{u}^{n}(X)` as no such root!

So be solving these equation for all the assets :math:`A` we obtain :math:`c` such that \
we have:

.. math:: 

    Ed^{s} = ML^{-1}L_{h} + c

.. automodule:: qoptimiza.simulation.simulation
    :members:
    :private-members: _chol, _get_random_unit_cov