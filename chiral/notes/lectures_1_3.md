# Lecture 1: Overview
- How do we study discrete objects using differential geometry? What is it good for, and how do you use these techniques in physics, machine learning, and signal processing?
- Overview of the course:
	- Link the mathematical perspective of shape (differential geometry) with the computational perspective (geometry processing)
	- Shapes are everywhere! Every time you're solving a constraint $f(x) = 0$, you have a manifold
- Many viewpoints of differential geometry that we won't have time to cover here
- Assignments: Derive geometric algorithms from first principles, then implement them
	- Discrete surfaces
	- Exterior calculus
	- Curvature
	- Smoothing
	- Direction field design
- Differential geometry is the language for talking about **local properties of shape**, and it can also be used to study **global properties of shape** by integrating these local quantities
	- Allows us to use the tools of calculus on arbitrary manifolds
- **Discrete** differential geometry (DDG): also a language to describe local properties of shape
	- No longer allowed to talk about derivatives, infinitesimals: everything must be expressed in terms of lengths and angles
	- Surprisingly little is lost in translation, and is the modern language of geometric computing
- **Goal**: Translate continuous differential geometry into a language suitable for computation
- Obtaining discrete definitions in DDG: 
	1. Write down several **equivalent** definitions in the smooth setting
	2. Apply each smooth definition to an object in the discrete setting
	3. Determine which properties are captures with each resulting **inequivalent** discrete definition. Often no single discrete definition captures all of the properties of its smooth counterpart.

	This is much like how we have different actions in lattice QCD: they all give the same thing in the $a\rightarrow 0$ limit, and they all have different properties at finite $a$. 
	
### Curvature
- Recall that a **parameterized curve** on a manifold is a smooth map $\gamma : [0, L]\rightarrow M$, where $L > 0$ is some real number. 
- A **discrete curve** is a piecewise linear parameterized curve, i.e. it is a sequence of vertices connected by straight line segments. Can also think of $\gamma : [0, L]\rightarrow M$ as an affine map which is linear between points $s_i$ and $s_{i + 1}$, and we'll write $\gamma_i := \gamma(s_i)$. 
- How do we approximate the tangent vector to a curve? Usually, we would define the tangent vector field as 
$$
T(s) := \frac{\gamma'(s)}{|\gamma(s)|} = \bigg|_{|\gamma(s)| = 1} \frac{d\gamma}{ds}
$$
where the bar assumes arc-length parameterization. 
- The **unit normal** to a curve can be expressed as a quarter-rotation $J$ of the unit tangent vector in the counter-clockwise direction:
$$
N(s) := J T(s)
$$
(there is also a definition in terms of derivatives). The **curvature** is defined as the inner product of $N$ and $T'$:
$$
\kappa(s) = \langle N(s), \frac{dT}{ds}\rangle = \left\langle J\frac{d\gamma}{ds}, \frac{d^2\gamma}{ds^2}\right\rangle
$$
- Two key ideas:
	1. Curvature is a second derivative.
	2. Curvature is a signed quantity. 
- How do we port this to the discrete setting? We can't just take derivatives, since they're either zero or infinity at any given point. How do we know if we've come up with a good definition?
	- Satisfies (some of) the same properties as smooth curvature.
	- Converges to smooth values as we refine our curve. 
	- Is it efficient to compute this quantity / solve equations with it?
- Initial definition of curvature: rate of change of $T(s)$ in the normal direction. Equivalently, measure the rate of change of the angle $\phi(s)$ the tangent makes with the horizontal, $\kappa = d\phi / ds$. 
	- We still can't differentiate this angle, but over any finite segment $[a, b]$ of the curve, the integral of the curvature is the chagne in angle:
	$$
	\int_a^b ds\, \kappa(s) = \phi(b) - \phi(a)
	$$
	This definition translates to discrete curves! We can get the total curvature along a segment by adding up all the curvatures. This definition is the **turning angle** definition. Let $\phi_{i, i + 1}$ be the angle between $\gamma_i$ and $\gamma_{i + 1}$. Then:
	$$
	\theta_i := \mathrm{angle}(\gamma_i - \gamma_{i - 1}, \gamma_{i + 1} - \gamma_i) \\
	\kappa_i^A := \theta_i\;\;\;\;\;\;\; (\mathrm{turning}\;\mathrm{angle})
	$$
	- Discrete definitions will often be **integrated quantities** in the smooth case rather than local quantities
- Other definitions: Fastest way to decrease the length of a curve is to move it in the normal direction, with speed proportional to the curvature. Intuition: shifting a flat curve doesn't change it's length, but shifting a circle by its normal will make it bigger
	- Consider an arbitrary change of the curve $\gamma$ by a perturbation $\eta : [0, L]\rightarrow\mathbb R^2$, with $\eta(0) = \eta(L) = 0$. Then:
	$$
	\frac{d}{d\epsilon}\bigg|_{\epsilon = 0} \mathrm{length}(\gamma + \epsilon\eta) = -\int_0^L ds\, \langle \eta(s), \kappa(s) N(s)\rangle
	$$
	- In the discrete setting, just take the gradient of the length of the curve with respect to the vertex positions. For example, for a line segment from $a$ to $b$, the quickest way to change the length of the curve is to extend it, i.e. make $b - a$ larger, which is $\nabla_b \ell$, where $\ell = |b - a|$ is the length. 
		- To find the motion which most quickly increases the total length $L$, just sum the contributions of each segment and take $\nabla_{\gamma_i} L$, where $L$ is the length. Since in the smooth setting, the gradient of length is equal to the curvature times the normal, we can immediately port this over to a discrete quantity. In terms of the angle $\theta_i$, one can work out that the **length variation** definition of curvature is:
		$$
			\kappa_i^B = 2\sin(\theta_i / 2)
		$$
- Note these are equivalent in the smooth setting, as $\theta_i\rightarrow 0$ (at any kink, the infinitesimal turning angle goes to zero, since for a smooth curve we can only take derivatives, not differences). 
- **Steiner formula**: This is closely related to the length variation approach:
$$
	\mathrm{length}(\gamma + \epsilon N) = \mathrm{length}(\gamma) - \epsilon\int_0^Lds\,\kappa(s)
$$
	- In discrete setting, normals aren't defined at vertices. We can offset individual edges along their normals, though (just translate each individual line segment). There's no explicit definition to connect these line segments, but we can vary the way we connect them, i.e. 
	<ol type="A">
		<li>With a circular arc.</li>
		<li>With a straight line.</li>
		<li>By extending the edges until they intersect.</li>
	</ol>
	The first two of these give us the $\kappa^A$ and $\kappa^B$ definitions! The $C$ case gives us another definition:
	$$
		\kappa^C_i = 2\tan(\theta_i / 2)
	$$
- Note that we arrived at $\kappa^A$ and $\kappa^B$ in multiple different ways: sometimes in DDG, the same definitions will arise over and over again. 
- Last definition: the osculating circle is the circle of radius $r(p)$ that best approximates the curve at point $p$, and the curvature is $\kappa(p) = 1 / r(p)$. 
	- We can approximate this and show that the radius of the best circle is $w_i / (2\sin\theta_i)$, where $w_i := \gamma_{i + 1} - \gamma_i|$. This gives the **osculating circle** definition of curvature:
	$$
	\kappa_i^D := \frac{2\sin\theta_i}{w_i}
	$$
- Moral of the story: pick the right tool for the job. For each definition you use, there will be tradeoffs

# Lecture 2a: Combinatorial Surfaces (Meshes)
- A **mesh** is a combinatorial surface. 
	- We'll make this precise with a simplicial complex
- Loosely speaking, a topological space tells us how points are connected and how close different points are, without talking about specifics of where they are in space. 
	- The discrete counterpart we'll be using is that of a combinatorial surface: it tells us what is connected to what else. 
- A subset $S\subset\mathbb R^n$ is **convex** if for every pair of points $p, q\in S$, the line segment between $p$ and $q$ is contained in $S$. 
- The **convex hull** $\mathrm{conv}(S)$ of a subset $S\subset\mathbb R^n$ is the smallest convex set containing $S$, or equivalently, the intersection of all convex sets containing $S$. 
- Roughly speaking, a $k$-simplex is a $k$-dimensional hypertriangle ($k = 0$ is a vertex, $k = 1$ is an edge, $k = 2$ is a triangle, $k = 3$ is a tetrahedron, and so on). Let's make this more precise.
	- A collection of points $p_0, p_1, ..., p_k$ are **affinely independent** if the vectors $v_i := p_i - p_0$ are linearly independent. 
		- This basically means they don't all lie on a straight line or in the same plane. If we shift the origin to $p_0$, it's just linear independence of the points.
	- A **k-simplex** is the convex hull of $k + 1$ affinely independent points, which we call its vertices.
		- For example, the convex hull of a point is a point. The convex hull of two points is a line, and the convex hull of 3 (affinely independent) points is a triangle.
		- If we don't assume the points are affinely independent, we get a degenerate definition. 
	- **Barycentric coordinates** help us to describe a simplex explicitly. For a 1-simplex between points $a$ and $b$< we cann parameterize the simplex as $p(t) = ta + (1 - t)b$. More generally, any point $p$ in a $k$-simplex $\sigma$ can be expressed as a non-negative weighted combination of the vertices, where the weights sum to 1. 
	$$
		\sigma = \left\{ \sum_{i = 0}^k t_i p_i \bigg| \sum_i^k t_i = 1, t_i\geq 0 \forall i \right\}
	$$
	This is called a **convex combination** of the points $p_i$. 
- *Definition*: the **standard n-simplex** is the collection of points:
	$$
	\sigma := \left\{ (x_0, ..., x_n)\in\mathbb R^{n + 1} \bigg| \sum_{i = 1}^n x_i = 1, x_i\geq 0, \forall i \right\}
	$$
	This is also called the probability simplex, and it looks like a hypertriangle in $\mathbb R^{n + 1}$. 
- A simplicial complex is a collection of simplices satisfying some specific properties that constrain how the simplices intersect.
	- A **face** of a simplex $\sigma$ is any simplex whose vertices are a subset of the vertices of $\sigma$. For a 2-simplex (triangle), a face is any edge or vertex of the triangle, so there are 3 $k = 1$ faces and 3 $k = 0$ faces. 
		- A face doesn't have to be a proper subset, so $\sigma$ is a face of itself. Formally, the empty set $\emptyset$ is also a face of $\sigma$.
	- *Definition*: A **(geometric) simplicial complex** is a collection $\mathcal K$ of simplices where:
		1. The intersection of any two simplices in $\mathcal K$ is a simplex in $\mathcal K$.
		2. Every face of every simplex in $\mathcal K$ is also in $\mathcal K$.
	- For example, the intersection of two triangles is an edge, so if we have two adjacent triangles their edge must be in $\mathcal K$. 
	- Given a simplicial complex, we'll work with it by *enumerating* each vertex, then using that to enumerate each edge, and so on. We can also instead assign a unique ID to each $k$-simplex, i.e. if I have vertices $0, 1$, then the edge between them is $\{0, 1\}$, or I can label it as edge $0$. 
	- The purpose of working with simplicial complices is that it tells us about the **connectivity** of the complex, not how it's sitting in space. It's a topological structure, not a geometric one. 
- *Definition*: Let $S$ be a collection of sets. If for each set $\sigma\in S$, all subsets of $\sigma$ are contained in $S$, then $S$ is an **abstract simplicial complex**. A set $\sigma\in S$ of size $k + 1$ is called an **abstract $k$-simplex**. 
	- This gives us a discrete analogue of a topological space
	- An example of this is a graph: a graph is a simplicial 1-complex. 
- Topological data analysis: Try to understand data in terms of connectivity. This leads into *persistent homology*, which is the idea of clustering points in space. The idea is to grow a disc around each point, and if the discs around different points inteserct you connect them with a simplex to make a simplicial complex. 
    - Track the "birth" and "death" of topological features that come into the emerging simplicial complex like connected components, holes, etc, and plot it as a function of disc size
    - Features which persist for a long time are likely signals of something, while those which are transient are likely noise. 
    - Materials science connection: Nakamura et al, "Persistent Homology and Many-Body Atomic Structure for Medium-Range Order in the Glass". 

#### Anatomy of a simplicial complex
Let $\mathcal K$ be a simplicial complex, and $A\subseteq \mathcal K$ a subset.
- The **closure** $CL(A)$ of a set $A$ is the smallest simplicial complex containing $A$ (equivalently, it's the intersection of all the simplicial subcomplices containing $A$)
- The (simplicial) **star** $St(A)$ of $A$ is the union of all simplices in $\mathcal K$ which contain $A$.
    - This generally differs from $Cl(A)$ because it doesn't contain the boundary of points around name, which is another object we'll define now:
- The **link** $Lk(A)$ of $A$ is defined as $Cl(St(A))\setminus St(Cl(A))$. 
- Some notation: for a simplicial 1-complex (graph), we often write $G = (V, E)$. For a simplicial 2-complex, we'll often write $\mathcal K = (V, E, F)$ where $F$ are the faces. 

#### Orientation
- To orient a 1-simplex $\{a, b\}$, we store it in a tuple $(a, b)$. This is the simplex from $a$ to $b$. We'll need this for the discrete exterior calculus. 
- Orientation of a 2-simplex: given by the winding order of the vertices
    - This is an equivalence class of ordering under cyclic permutations, i.e. the oriented 2-simplex $(a, b, c), (b, c, a), (c, a, b)$ are all the same. 
- **Orientation of a $k$-simplex**: An oriented $k$-simplex is an ordered tuple of $k + 1$ elements (vertices), defined up to even permutation. 
    - This means we always have two orientations: even or odd permutations of the vertices. Even permutations of $(0, ..., k)$ have **positive** orientation, and odd permutations of this are negative. 
- What is the orientation of a single vertex? There's only a single orientation (positive)
- Ex: 3-simplex: Even elements are $(1, 2, 3, 4), (2, 3, 1, 4), ...$, and odd elements are $(1, 2, 4, 3), (4, 2, 3, 1)$, etc. 
- An **oriented** of a simplex is an ordering of  of its vertices up to even permutation.
    - One can specify the oriented simplex via a representative. 
- An **oriented simplicial complex** is a simplicial complex where each simplex is given an ordering. 
    - Note that these orderings don't have to be consistent! However, we'll often need them to be for certain extensions
- *Definition*: Two distinct oriented simplices have the same **relative orientation** if the two (maximal) faces of their intersection have *opposite* orientation.
    - For example, the 2-simplices $(a, c, b)$ and $(c, d, b)$ have the same relative orientation (draw them out), because $(c, b) = -(b, c)$. 
- You cannot always assign a consistent orientation for an arbitrary simplicial complex if there's too much twisting going on, i.e. a Mobius band. 

<!-- # Lecture 2b: Simplicial Manifolds -->
#### Miscellaneous
- Musical isomorphisms: allow us to go between a vector space $V$ and its dual $V^*$, and is especially useful to get from the tangent bundle $TM$ to the cotangent bundle $T^* M$. Denote a basis for $V$ as $\{\partial_\mu\}$, and a basis for $V'$ as $\{dx^\mu\}$
  - Note this notation is very degenerate, as $\{\partial_\mu\}$ can often denote a frame for the tangent bundle $TM$, and likewise for $\{dx^\mu\}$. 
  - Recall that in the geometric setting, a vector in $T_p M$ is parameterized as $v^\mu \partial_\mu$, and a covector is parameterized as $w_\mu dx^\mu$. The musical isomorphisms are:
  $$
	\flat : TM\rightarrow \Omega^1(M), v^\mu \partial_\mu\rightarrow v_\mu dx^\mu
  $$
  where it is understood that the components $v^\mu = v_\mu$