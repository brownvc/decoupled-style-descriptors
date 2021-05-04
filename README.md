# Generating Handwriting via Decoupled Style Descriptors

[Atsunobu Kotani](http://www.atsunobukotani.com/research/), [Stefanie Tellex](http://cs.brown.edu/people/stellex/), [James Tompkin](www.jamestompkin.com)

ECCV 2020

http://dsd.cs.brown.edu/

![High-level overview of approach.](sample.gif)

We synthesize handwriting (bottom) in a target style (top) via learned spaces of style and content,
and can exploit any available reference samples (middle) to improve output quality.

## Code release coming soon!
Please watch and star this repo to be notified : )  
*Our apologies for the delay; COVID-19 has caused some difficulties.*



# BRUSH dataset

This is the BRUSH dataset (BRown University Stylus Handwriting) from the paper "[Generating Handwriting via Decoupled Style Descriptors](http://dsd.cs.brown.edu/)" by Atsunobu Kotani, Stefanie Tellex, James Tompkin from Brown University, presented at European Conference on Computer Vision (ECCV) 2020. This dataset contains 27,649 online handwriting samples, and the total of 170 writers contributed to create this dataset. Every sequence is labeled with characters (i.e. users can identify what character a point in a sequence corresponds with.

The BRUSH dataset can be downloaded from
[this link](https://drive.google.com/drive/folders/1wUgNsBebpIZJEATlduB8LjGKEHOWtv2o?usp=sharing) (compressed ZIP 566.6MB).


## Terms of Use
The BRUSH dataset may only be used for non-commercial research purposes.
Anyone wanting to use it for other purposes should contact [Prof. James Tompkin](www.jamestompkin.com).
If you publish materials based on this database, we request that you please include a reference to our paper.

```
@inproceedings{kotani2020generating,
  title={Generating Handwriting via Decoupled Style Descriptors},
  author={Kotani, Atsunobu and Tellex, Stefanie and Tompkin, James},
  booktitle={European Conference on Computer Vision},
  pages={764--780},
  year={2020},
  organization={Springer}
}
```


## Data specification

Each folder contains drawings by the same writer, and each file is compressed with Python 3.8.5 pickle as it follows.
```{python}
import pickle
with open("BRUSH/{writer_id}/{drawing_id}", 'rb') as f:
  [sentence, drawing, label] = pickle.load(f)
```
Each file is comprised of the following data.
  1.  **Target sentence** -- a text of length (M).

      If a sample is a drawing of "hello world", for instance, this value would be "hello world" and
      M=11 as it is.

  2.  **Original drawing** -- a 2D array of size (N, 3).

      We asked participants to write specific sentences in a box of 120x748 pixels, with a suggested
      baseline at 100 pixels from the top, 20 pixels from the bottom. As every drawing has a different
      sequence length, N can vary, and each point has 3 values; the first two are (x, y) coordinates
      and the last value is a binary end-of-stroke (eos) flag. If this value were 1, it indicates that
      the current stroke (i.e. curve) ends there, instead of being connected with the next point, if
      it exists.

  3.  **Original character label** -- a 2D array of size (N, M).

      For every point in a sequence, this variable provides an one-hot vector of size (M) to identify
      the corresponding character.

As different writers used different kinds of a stylus to produce their data, the sample frequency (i.e.
the number of sampled points per second) varies per writer. For our original drawing data, we sampled
points at every 10ms (0.01s) by interpolating between points.

![Original drawing](samples/original.png)
In this example drawing, N=689 and M=19 (i.e. 'qualms politics ;[A'). Different colors indicate different character labels.

## Additional formats

We also included few other versions of resampled data. Both versions contain 5x rescaled drawings and
sacrifice temporal information in exchange for a consistence distance between points in a sequence.
  1.  **"BRUSH/{writer_id}/{drawing_id}_resample20"**

      Points are resampled in a way that they are distant from their previous point in a stroke sequence
      at maximum of 20 pixels.
      ![resampled 20](samples/20.png)
      In this resampled drawing, N=449 and M=19 (i.e. 'qualms politics ;[A').


  2.  **"BRUSH/{writer_id}/{drawing_id}_resample25"**

      Points are resampled in a way that they are distant from their previous point in a stroke sequence
      at maximum of 25 pixels.
      ![resampled 20](samples/25.png)
      In this resampled drawing, N=360 and M=19 (i.e. 'qualms politics ;[A').

