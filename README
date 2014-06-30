# liblistrankmf: An implementation of ListRank-MF
Copyright (c) 2014 Gabriel Poesia <gabriel.poesia at gmail.com>

## Description

Implementation of the ListRank-MF algorithm presented in the following paper:

Shi, Yue, Martha Larson, and Alan Hanjalic. 
"List-wise learning to rank with matrix factorization for collaborative filtering."
Proceedings of the fourth ACM conference on Recommender systems. ACM, 2010.

ListRank-FM is a List-wise Learning to Rank algorithm for Recommender Systems.
It takes as input a matrix with known ratings that users gave to some items,
and produces as output two matrices, one for users and one for items.
Each of these matrices has a (latent) feature vector for each user or item.
The predicted score of an item for a user is then given by the inner product
of their feature vectors. These scores have no intrinsic meaning, and
are optimized for ranking, not rating prediction.
They only serve the purpose of sorting the items by (predicted) preference
for each user. Using this, one can produce recommendations.

## Example 

An example program is given in `example.cpp`. It reads a matrix from 
standard input with each line containing the known ratings for one user.
The ratings are given in the sparse format ``item_id:rating``. For example,
1:2 means the user gave a rating of 2 to item 1. For example, the following
is a valid input file:

```
1:2 3:5 4:5
2:3 6:1 3:2
5:5 6:5
6:2 1:5 2:5
```

This input has 4 users. The first, second and fourth have 3 known ratings each,
and the third has two. The example program will use this file as a training
set for ListRank-MF and output another file in the same format, but with
the predicted scores for each user with respect to each item. Running the 
program given the input above gives the output:

```
1:0.851643 3:1.08888 4:1.11222 2:1.46886 6:0.197725 5:0.885626
1:0.593794 3:0.524859 4:0.713789 2:0.915647 6:0.123612 5:0.707199
1:0.71775 3:0.810566 4:0.980269 2:1.37975 6:0.285633 5:0.824209
1:0.746979 3:0.688814 4:0.974642 2:1.06619 6:0.13357 5:0.698586
```

Using this output, we can infer user 1 would prefer to see item 2 (which
has a score of 1.11222 for him) than item 6 (which has a lower predicted score
of 0.197725). Note that the predicted scores tend to be consistent (with
respect to their relative order) with the observed ratings given in the
input. This happens for all users in this example.

## License

This library is released under the MIT license. In practice, this means
you can use it freely provided that you keep the copyright notice.

## Last words

If you find this library useful for any purpose, I'd be very pleased 
to hear about that! I'll appreciate if you send me an email simply
telling me what you are using liblistrankmf for (research, personal
projects, etc).
