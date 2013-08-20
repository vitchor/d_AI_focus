d_AI_focus
==========
dyfocus using AI

<b>1st Step: Tutorial.<b/>
  - Watch this amazing tutorial: http://www.youtube.com/watch?v=4ONBVNm3isI

<b>2nd Step: Set up development environment.<b/>
  1. Install iPython: $ sudo easy_install ipython[all]
  2. Install Macport using dmg: http://www.macports.org/install.php
  3. Close/Open terminal.
  4. Install Math pre-requisits (takes a lot of time): $ sudo port install py27-numpy py27-scipy py27-matplotlib py27-ipython +notebook py27-pandas py27-sympy py27-nose
  5. Select installed frameworks: 
    - $ sudo port select --set python python27
    - $ sudo port select --set sphinx py27-sphinx
    - $ sudo port select --set ipython ipython27
    - $ sudo port select --set cython cython27
    - $ sudo port select --set py-sympy py27-sympy
  6. Install scikit: $ sudo port install py27-scikit-learn
  7. Close/Open terminal.
  8. Download IA framework:$ git clone git://github.com/jakevdp/sklearn_pycon2013.git
  9. Go into the IA's notebooks folder:$ cd sklearn_pycon2013/notebooks
  10. Start and open tutorial server: $ ipython notebook

<b>3rd Step: Run our ward script to get a piece of action<b/>
- In your terminal, cd to the "development" folder.
- Run the following: $ python ward.py input/sun_1.jpeg input/sun_2.jpeg 0.1 3
 
<b>4th Step: Edit this tutorial and make it better. :)<b/>

==========
<b>IMPORTANT:<b/>
fix jpeg bug, install lib:
http://ethan.tira-thompson.com/Mac_OS_X_Ports.html

==========
<b>Learning Material: SciKit First steps<b/>

Matplot's plots example page:
http://matplotlib.org/gallery.html

Connectivity graph of an image:
http://scikit-learn.org/stable/modules/feature_extraction.html

Abstract model of all Machine Learning methods:
http://peekaboo-vision.blogspot.com.br/2013/01/machine-learning-cheat-sheet-for-scikit.html

Clustering examples (DBScan is cool):
http://scikit-learn.org/stable/auto_examples/index.html#clustering

ML Lectures:
https://class.coursera.org/ml/lecture/preview

==========
<b>Learning Material: Image Clustering<b/>

Image to 2D array: http://wiki.scipy.org/Cookbook/Matplotlib/LoadImage

histogram 
http://scikit-image.org/docs/dev/auto_examples/plot_hog.html



