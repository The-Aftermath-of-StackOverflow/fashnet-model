# fashnet-model
<hr>

## What is the project about ?
Upon adding a new item to the database, the classification model 'cloth_cat' is pinged to supply the new_entry with descriptors which are used to index the item closer to similar items.


## Architecture Used
Each Dense Layer is followed by a Dropout layer, to regularize the network.
The model summary is as follows : 

## Libraries used for Modelling
<ol> Keras : to implement the sequential layers along with dropout layers.
<li> pickle : to serialize and pack the model for later use 
</ol>

## Arbitary rule
To reduce latency, vanilla neural networks are used. This limits the input vector size to a fixed number and hence all further additions have to be of the size 28,28pixels. <br>
To avoid having a fixed input length, we can make use of subsequenct conv_layers but that takes a toll on performance, requiring more memory and time simply due to the larger number of calculations.

## Scope of Improvement
<ul>
<li> Extract more descriptors from image.
<li> Work with multi channel inputs.
</ul>
<hr>
