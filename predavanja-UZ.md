# Image processing

## Image formation

<table style="border-collapse: collapse; border: none;">
<tr>
<td valign="top" width="50%" style="border: none;">

+ pinhole camera: box with a small aperture, image is turned upside down
+ $FOW = 2 \times \varphi$ &rarr; angular measure of space percieved by the camera:
    + $\varphi = \arctan(\frac{d}{2f})$
    + small $f$ &rarr; wide-angle image = large $FOW$
    + large $f$ &rarr; telescopic image = small $FOW$
</td>
<td valign="top" width="50%" style="border: none;">

![Optional Text](uz-slike/fow.png)
<!-- Replace `URL_to_Image` with the actual URL of your image -->

</td>
</tr>
</table>

+ effects of **aperture size**:
    + too large &rarr; multiplle directions averaging, blurred image
    + too small &rarr; light starts diffracting, blurred image
    + generally, a small number of rays hit the film, which results in a dark image &rarr; add a lens
+ the **lens** focuses light to film &rarr; the rays that travel through the center do not refract, points at particular distance remain in-focus, others are blurred
    + thin lens &rarr; points at different depths get focused on different depths of image plane
    + depth of field = distance between image planes at which the blutting effect is suffieiently small
    + small aperture increases depth of field
+ **chromatic aberration** = different wave-lengths refract at different angles and focus at different distances
+ **spherical aberration** = spherical lenses do not dlight perfectly - rays closer to lens edge focus closer than those at center
+ vignetting, radial distortion

## Color perception

For digital images, we use a matrix of sensors and quantise light into intensity levels.

In digital cameras, in classical design, we cannot read of R, G, and B channel at a single pixel. Most cameras use **Bayer filer** (CCD based - reads out charge serially and digitizes), which is a color filter array for arranging RGB color filters on a square grid. It has the twice as many G as B and R, because luminance is mostly determined by the green values, and human eye is much more sensitive to changes in intensity than color.
**Foveon X3** is CMOS based sensor (digitization on each cell seperately), and its based on the fact that RGB colors penetrate the silicon at different depths. It gives better image quality.

## Image processing 1

+ RGB to grayscale: $\frac{I(x,y,1) + I(x,y,2) + I(x,y,3)}{3}$
    + at each coordinate $[x,y]$, we have 3 gray-scale values (RGB)
+ binary images - only 1 or 0 (foreground/background), used in machine vision
    + we get binary images from grayscale images using **thresholding**

#### Image thresholding
+ transformation of a grayscale to binary
+ we can apply single or multiple thresholds, or a classifier
+ generally, a difficult problem &rarr; what threshold to pick?
+ threshold $T$ should **minimize** intensity variances within classes seperated by it, and **maximize** the betweeen class variance
 
**Otsu's algorithm:**
For a threshold $T$:
1. seperate the pixels into 2 groups by $T$
2. for each group, get an average intensity, and calculate the between class variance
3. select the $T*$, that maximizes the variance: $T* = \argmax_T  [\sigma^2_{between}(T)]$

**Generalization of Otsu:**
+ formulate the problem as fitting 2 Gaussians to the histogram with priors on mizture weights and variances
+ efficiently computed by a single pass through the histogram
+ outperforms all single-pass and all DL algorithms

**Local binarization:**
+ estimate a local threshold in the neighborhood $W$: $T_W = \mu_W + k \cdot \sigma_W$
+ $k \in [-1,1]$, set by user
+ calculates $T$ seperately for each pixel

#### Cleaning the image

+ thresholded image still includes noise
+ morphological operators: remove isolated points and small structures and fill the holes
+ structural element (SE) can be any size:
    + **fit:** all 1 pixels in SE cover 1 pixels in iamge
    + **hit:** any 1 pizels in SE cover 1 pixels in image

**Erosion:**
+ reduce the size of structures (remove bridges, noise...) &rarr; less white
$$
g(x,y) =
\begin{cases} 
1 \quad \text{if } s \textbf{ \textcolor{blue}{fits} } f \\
0 \quad \text{otherwise}
\end{cases}
$$

**Dilation:**
+ increases the size of structures, fills holes in regions &rarr; more white
$$
g(x,y) =
\begin{cases} 
1 \quad \text{if } s \textbf{ \textcolor{blue}{hits} } f \\
0 \quad \text{otherwise}
\end{cases}
$$

**Opening:**
+ erosion, dilation
+ removes small objects, preserves rough shape
+ can filter out structures by selecting the size of structuring element

**Closing:**
+ dilation, erosion
+ fill holes, perserves original shape
+ size of SE determines the max size of holes that will be filled

#### Labelling regions

Goalis to find seperate connected regions. Connectivity determinsed which pixels are considered neighbors.

**Sequential connected components:**
+ process image from left to right, from top to bottom:
    + if current value is 1:
        + if only one nrighbor is 1, copy its label
        + if both are 1 and have the same label, copy the label
        + if they have different labels: copy left label, update table of equivalent labels
        + else, form a new label
+ relabel with the smallest equivalent labels

#### Region descriptors

We can describe an area by perimeter, compactness, centroid... **Ideal descriptor** maps two images of the same object close-by in feature space, and two images of different objects to points far between each other.

## Image processing 2

#### Filtering

+ noise reduction, image restoration
+ structure extraction/enhancement
+ **Gaussian noise:** the intensity variation sampled from a normal distribution
+ **Salt and pepper noise:** random black and white descriptors
+ **Impulse noise:** random occurrence of white dots

**Gaussian filter:**
+ Gaussian noise = the intensity variation sampled from a normal distribution
+ assumptions: pixels are similar to their neighboring pixels, the noise is independent among pixels
+ remove the noise by computing an averafe of pixel intensities in a pixel's immediate neighborhood

**Correlation filtering:**
+ replace image intensity with a weighted sum of a window centered at that pixel
+ the weights in the linear combination are prescribed by the filter's kernel

**Convolution filtering:** 
+ flip the filter in both dimensions, apply cross-correlation
+ shift invariant
+ linear
+ commutative ($f * g = g * f$), associative ($(f*g)*h = f*(g*h)$) &rarr; applicatiuon of multiple filters is equal to aplication of a single filter, identity (unit impulse), derivative
+ for a symmetric filter, correlation = convolution

**Boundary conditions:**
+ kernel exceeds image boundaries at thye edge
+ crop, bend image, replicate edges, mirror image

**Gaussian kernel:**
+ instead of using uniform weights, pixels closer to the center should have higher weight
+ variance determines the extent of smoothing
+ half size of kernel $= 3\sigma$

**Convolution and spectrum:**
+ convolution of 2 functions in image space is equivalent to the product of their corresponding Fourier transforms &rarr;
 $F(f*g) = F(f) \odot F(g)$
 + convolution manipulates image spectrum (enhances/suppresses frequency bands)
+ Fourier transform = a signal is represented as a sum of sines/cosines of various frequencies

**Removing noise:**
+ noise = adding high frequencies &rarr; to remove them we apply a low-band pass filter (allows low-frequency signals to pass through while reducing the amplitude of frequencies higher than some threshold)
+ a Gaussian manitains compact support in image and frequency space &rarr; appropriate low-band pass filter

**Sharpening filter:**
+ linear filter
+ enhances differences by local averaging

**Filtering as template matching:**
+ apply correlation with template &rarr; dot product = measure of similarity
+ correlation map
+ issue with scaling &rarr; instead of scaling the template, we scale the image

**Reducing an image:**
+ we can't remove every second... pixel &rarr; aliasing (distortion of the signal, because it's not sampled accurately)
+ Nyquist theorem: if we want to reconstruct all frequencies up to $f$, we have to sample the signal by at lease a frequency, equal to $2f$
+ solution: remove the high frequencies that cannot be reconstructed by blurring, then subsample &rarr; Gaussian pyramid
+ reason fo size reduction: Gaussian is a low-band pass filter, so we gerte a redundant representation of the smoothed image &rarr; no need to store it in full resolution

**Median filter:**
+ nonlinear filter
+ replace the pixel intensity by a median of intensities within a small patch
+ doesn't add new gray-levels into the image, removes outliers &rarr; good for impulse and salt&pepper noise removal

#### Color

Light:
+ EM radiaton composed of several frequencies
+ properties described by its spectrum (how much of each frequency is present)

Human color perception:
+ cones &rarr; cells that react differently to different wavelenghts (RGB)
+ rods &rarr; for intensity

**Additive mixture model:** RGB colors added to black (white in the middle), monitors, projectors
**Subtractive model:** cyann, yellow, violet pigment added to white paper (black in the middle) &rarr; pigments remove color, printers, photographic film

Color space is a unique color specification (for reproduction), a new color is a weighted sum of primaries. In uniform color space, the perceptual differences between colors are uniform across the entire space.

#### Color description by using histograms

Image histogram records the frequency of intensity levels. In a color histogram, each pixel is a point in 3D space (RGB), where $H(R,G,B)=$ number of pixels with color $[R,G,B]$. It's a robust representation of images.
Intensity is contained in each color channel. Multiplying a color by a scalar changes the intensity, not the hue. So we can normalize a color by its intensity: $I = R+ G+B$, $r = \frac{R}{I}$, $r+g+b=1$.

If we want to compare images by their descriptors - histograms, we have to measure a distance between histograms. We can do that with: Euclidean distance (differences in histogram cells), pdf similarity (2 probability density functions - Chi-squared, Hellinger...).

# Edge detection

Goal is to map an image from 2D grayscale intensity pixel array into a set of binary curves and lines.

Anything that appears as an edge, constitutes an edge &rarr; local texture, shape changes, discontinuity of depth, shadows... Edge presence is strongly correlated with the local intensity changes &rarr; **derivative** measures a local intensity change.

#### Image derivatives

+ horizontal derivative: $[-1,1]$
+ vertical derivative: $[-1,1]^T$
+ gradient $\Delta f = [\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}]$ &rarr; points in direction of greatest intensity change
+ smooth the image and then search for edges (remove the noise) &rarr; maxima of $\frac{\partial}{\partial x}(I * G) = I * (\frac{\partial}{\partial x}G)$
+ $\sigma$ is the scale/width of a Gaussian kernel, that determines the extent of smoothing &rarr; which edges will be removed
    + large kernel: detect edges on a larger scale
    + small kernel: detect edges on a smaller scale

#### From derivatives to edge detection

Approach: find strong gradients + post process.

**Optimal edge detector:**
+ good detection: minimizes probability of false positives (edges caused by noise) and false negatives (missing true edges)
+ good localization: detected edges are close to the location of the true edges
+ specificity: returns only a single point per true edge (minimize number of local maxima around true edge)

**Canny edge detector:**
+ most popular
+ a step function + Gaussian noise
+ first derivative of a Gaussian well approximates an operator that optimizes a tradeoff between signal-to-noise ratio and localization on the specified theoretical edge model
+ method:
    + filter (convolve) image by a derivative of Gaussian (smooth and enhance)
    + calculate the gradient magnitude and orientation
    + thin potential edges to a single pixel thickness &rarr; **non-maxima suppression**
        + for each pizel check, if it's a local maximum along its gradient direction
        + only local maxima remain
    + select sequences of connected pixels, that are likely an edge &rarr; **hysteresis thresholding**
        + apply 2 thresholds
        + start tracing a line only at pixels above $k_{high}$ and continue tracing if pixel exceed $k_{low}$

#### Edge detection by parametric models

Line fitting - many scenes are composed of straight lines. Problems with line fitting: nopisy edges (which points correspond to which lines), some parts of lines not detected, noisy orientation... We use line fitting by voting for parameters with Hough transform. For eache dge point, compute parameters of all possible lines passing through it, and for each set of parameters cast a vote. Then select the lines that recieve enough votes.

**Hough space:**
+ a line in image corresponds to a point in Hough space
+ for a point $(x,y)$, find all $(m,b)$ for which holds $y=mx+b$
+ a point in image space corresponds to a line in Hough space &rarr; if 2 lines intersect in Hough space at $(m_{opt}, b_{opt})$, then they lie on the same line $y = m_{opt}x + b_{opt}$ in the image space
+ issue with Cartesian $(m,b)$: we get infinite values for vertical lines &rarr; we use a sinusoid in Hough space to represent a point in the image: $x\cos\theta -y\sin\theta = d$

**Hough transform:**
1. initialize $H[d,\theta]=0$ (*accumulator array*)
2. for each edge point $(x,y)$ in image:
    &nbsp;&nbsp;&nbsp;&nbsp; for  $\theta=0$ to $180$ do:
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$d=x\cos\theta -y\sin\theta$
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$H[d,\theta]+=1$
3. find local maxima $\{ d^{i}_{\text{opt}}, \theta^{i}_{\text{opt}} \}_{i=1}^{N}$ in accumulator array $H$
4. detected line is defined by $d^{i}_{\text{opt}} = x \cos \theta^{i}_{\text{opt}} - \sin \theta^{i}_{\text{opt}}$

Problem with noise in Hough transform: random points still form some local maxima in the accumulator array. The solution is that instead of checking each theta, we use gradient direction at $(x,y)$ as $\theta$. And instead of assigning the same vote for everyone, we use magnitude: $H(d,\theta) += m(x,y)$.

We can also use Hough transform for circles, where we draw a circle with a predifined radius in Hough space for every point in image space. The intersection of the circles in Hough space corresponds to the center of the circle in image space. For an unknown radius, we need a 3D Hough space, where we map cones.

**Generalized Hough transform (GHT) for shape-based models:**
+ define the shape model by edge points and a reference point
+ model learning: for each edge point use its gradient orientation $\theta$ to index into the table and use the displacement vectors $\vec{r}$ to cast a vote for the center

Advices for Hough transform: minimize irrelevant responses, appropriately discretize the parametric space, vote for neighboring cells as well (cast a vote with a Gaussian), use gradient direction

**Pros:** each point processed independently (robustness, parallelizable), robustness to noise, can detect multiple instances of a single model in one pass.
**Cons:** time complexity increases exponentially with number of free parameters, some shapes may generate false local maxima, choosing the accumulator cell size is application dependent.

# Fitting parametric models

+ transformation: $X'_i = f(X_i;\vec{p})$
+ $X_i = [x_i, y_i]^T$
+ $\vec{p}=[p_1,...,p_m]$

Use cases:
+ given a set of correspondences, what are the parameters of the transformation?
+ assuming the transformation can be well approximated by $f(x;p)$, what are the best parameter values for $p$? &rarr; those that minimize the projection error
+ used for mapping 2D into 3D

### Least squares: Line fitting

**Least squares:**
+ data: $\{(x_1,y_1),...,(x_N,y_N)\}$
+ line equation: $y=f(x;p)=xp_1+p_2$
+ projection error at i-th correspondence: $\varepsilon_i=f(x_i;p)-y_i$
+ cost function: $E(p)=\sum_{i=1}^N\varepsilon_i^2$
+ best parameters: $\hat{p} = \argmin_p E(p)$

**Minimizing least squares:** $p=A^\dagger b$

**Weighted least squares:**
+ weighted cost function: $E(p)=\sum_{i=1}^N w_i \varepsilon_i^2$
+ can be used for nonlinear/robust least squares

**Minimizing weighted least squares:** $p=(A^TWA)^{-1}A^TWb$

**Constrained least squares:**
+ cost function: $E(p)=\sum_{i=1}^N  ||\varepsilon_i||^2$
+ trivial solution is $p=0$, nontrivial solution is obtained by constraint $||p||^2=1$

**Minimizing constrained least squares:**
+ $A^TAp=\lambda p$
+ the solution is the eigenvector of $(A^TA)$, corresponding to the smallest eigenvalue, which is the eigenvector, corresponding to the smallest eigenvalue of $A$
+ SVD decomposition

We often use **nonlinear error/cost functions**, which cannot be minimized analytically in a closed form. Then we use gradient descend, Newton method, Gauss-Newton method...

### Outliers, RANSAC

Large disagreements in only a few points - outliers, cause falioure of the least squares based methods. The direction, localization and recognition have to operate in significantly noisy data. In some cases, more than a half of data is expected to be outliers. Standard models for robust estimation can rarely deal with such large proportion of outliers.

RANSAC - random sample consensus algorithm is very popular due to its generality and simplicity. It can deal with large portions of outliers. We find a model with strong support by randomly sampling potential models.

If we want to fit a line to points, we randomly sample 2 points and connect a line through them. Then we count the number of inliers - all points whose $\varepsilon_i$ is lower than some threshold. We repeat that over $N$ iterations, or until the number of inliers becomes big enough.

**RANSAC algorithm:**
1. randomly select the smallest group of correspondences, from which we can estimate the parameters of our model
2. fit a parametric model $\hat{p}$ to the selected correspondences
3. project all other points and count the number of inliers

The model parameters $\hat{p}_{opt}$ maximize the number of inliers.

**The choice of parameters:**
+ how many correspondences? as many as the model parameters
+ threshold? choose $t$ such that the probability that an inlier falls below $t$ is $p_w$, usually $p_w=0.95$, if we have a Gaussian: $t=2\sigma$
+ number of iterations? choose $N$ such that the probability $p$ of drawing a sample with all inliers at least once is high enough
    + $e$ = proportion of outliers (probability of selecting an outlier at random)
    + probability of choosing a single inlier = $1-e$
    + we have $s$ correspondences so probability of all-inlier sample = $(1-e)^s$
    + probability that at least one in $s$ is not an inlier = $[1-(1-e)^s]$
    + $1-p = [1-(1-e)^s]^N \rightarrow N = \frac{\log(1-p)}{\log(1-(1-e)^s)}$

We can **improve RANSAC** by applying least squares to the inliers. After computing the best inliers, we use least squares to fit a new model to all the inliers. This makes the final model more accurate.

If we have multiple instances of our model (multiple lines for example), we apply voting to RANSAC. If we have multiple models, we apply model selection.

**Pros:** simple, applicable, often used in practice
**Cons:** requires setting some parameters, potentially a lot of iterations, fails at small number of inliers

# Local features

# Camera geometry

# Multiple-view geometry

# Recognition & Detecion