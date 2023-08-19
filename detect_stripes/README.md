# Find Category of image
<hr>

<h3> The model has been trained on the given categories </h3>
<ol>
<li> 'animal'
<li> 'cartoon'
<li> 'chevron'
<li> 'floral'
<li> 'geometry'
<li> 'houndstooth'
<li> 'ikat'
<li> 'letter_numb'
<li> 'OTHER'
<li> 'plain'
<li> 'polka dot'
<li> 'scales'
<li> 'skull'
<li> 'squares'
<li> 'stars'
<li> 'stripes'
<li> 'tribal' </ol>

<hr>

<h3> model.py</h3>

Train a CNN for the given 17 classes. The training accuracy is 96.32% and validation accuracy is 95.03%. <br>
The CNN has two convulational layers, each followed by a respective max_pool layer. For the Fully connected part, we make use of 2layers which brings up the majority of the parameters of the total 815k parameters. <br>

After training the model, save the model and its weight.

<h3> model_eval.py </h3>
Predict the category of given image.

<h3> What can the model also do ?</h3>
Predict similar images, 