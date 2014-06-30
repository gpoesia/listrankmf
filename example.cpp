/*
   The MIT License (MIT)

   Copyright (c) 2014 Gabriel Poesia <gabriel.poesia at gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

using namespace std;

#include "./listrankmf.h"

vector<vector<pair<int, double>>> read_input(
        istream &input, vector<int> &ids_to_movies) {
    vector<vector<pair<int, double>>> ratings;
    map<int, int> movie_ids;

    string line;
    while (getline(input, line)) {
        ratings.resize(ratings.size() + 1);

        istringstream iss(line);

        string rating;

        while (iss >> rating) {
            unsigned movie_id = 0, rating_value = 0;

            int i = 0;

            while (rating[i] != ':') {
                movie_id = 10*movie_id + (rating[i] - '0');
                i++;
            }

            rating_value = rating[i+1] - '0';

            unsigned movie_index = 0;

            if (movie_ids.count(movie_id) == 0) {
                movie_index = ids_to_movies.size();
                ids_to_movies.push_back(movie_id);
                movie_ids[movie_id] = movie_index;
            } else {
                movie_index = movie_ids[movie_id];
            }

            ratings.back().push_back(make_pair(movie_index, rating_value));
        }
    }

    return ratings;
}

void print_output(ostream &output,
        vector<vector<double> > &users_features,
        vector<vector<double> > &movies_features,
        const vector<int> &ids_to_movies) {
    for (unsigned i = 0; i < users_features.size(); i++) {
        output << ids_to_movies[0] << ":"
            << predict_score(users_features[i], movies_features[0]);

        for (unsigned j = 1; j < movies_features.size(); j++) {
            output << " " << ids_to_movies[j] << ":"
                << predict_score(users_features[i], movies_features[j]);
        }

        output << '\n';
    }
}

int main() {
    vector<int> ids_to_movies;
    vector<vector<pair<int, double>>> ratings_matrix;
    ratings_matrix = read_input(cin, ids_to_movies);

    vector<vector<double>> movies_features;
    vector<vector<double>> users_features;

    list_rank_mf(ratings_matrix, users_features, movies_features);
    print_output(cout, users_features, movies_features, ids_to_movies);

    return 0;
}
