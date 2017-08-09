//
// Created by Tianzi Harrison on 7/30/17.
// http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
// https://www.tutorialspoint.com/cpp_standard_library/vector.htm
//

#include <iostream>
#include <math.h>
//#include<numeric>
#include <vector>
#include "classifier.h"


#define _USE_MATH_DEFINES

using namespace std;


/**
 * Initializes GNB
 */
GNB::GNB()
: stats(3, vector<vector<double>> (4, vector<double> (2, 0.0)))
{}

GNB::~GNB() {}

double GNB::find_mean(vector<double> v) {
    //return accumulate(v.begin(), v.end(), 0.0) / v.size();

    double sum = 0.0;
    for(int i = 0; i < v.size(); i++)
        sum += v[i];
    return (sum / v.size());
}

double GNB::find_stdev(vector<double> v) {

/*    size_t n = v.size();
    vector<double> diff(n);
    double mu = find_mean(v);
    transform(v.begin(), v.end(), diff.begin(), [mu](double x) { return x - mu; });
    double sq_sum = inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    return sqrt(sq_sum / (n-1));*/

    double mu = find_mean(v);
    double temp = 0.0;

    for(int i = 0; i < v.size(); i++){
        temp += (v[i] - mu) * (v[i] - mu) ;
    };
    return sqrt( temp / (v.size() - 1));

}

vector<vector<vector<double>>> GNB::train(vector<vector<double>> data, vector<string> labels) {

    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                  [3.5, 0.1, 5.9, -0.02],
                  [8.0, -0.3, 3.0, 2.2],
                  ...
              ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".

        update stats [mu, stdev] for s, d, s_dot, d_dot for each label
    */

    vector<vector<double>> ll, kl, rl; // ? by 4

    for (int i = 0; i < labels.size(); i ++){
        switch (labels[i].at(0))
        {
            case 'l':
                ll.push_back(data[i]);
                break;
            case 'k':
                kl.push_back(data[i]);
                break;
            case 'r':
                rl.push_back(data[i]);
                break;
        }
    };

    vector<vector<vector<double>>> l;
    l.push_back({ll}); l.push_back({kl}); l.push_back({rl});
    for(int i = 0; i < l.size(); i ++){
        size_t n = l[i].size();
        for(int j = 0; j < 4; j ++){
            vector<double> temp;
            for(int k = 0; k < n; k ++){
                temp.push_back(l[i][k][j]);
            }
            double mu = find_mean(temp);
            double stdev = find_stdev(temp);
            stats[i][j].assign({mu, stdev});
        }
    }

    return l;
}

string GNB::predict(vector<double> v) {
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
        # TODO - complete this
    */

    vector<double> probabilities;

    for(int i = 0; i < 3; i ++){
        probabilities.push_back(1.0);
        for (int j = 0; j < 4; j ++){
            double mu = this->stats[i][j][0];
            double stdev = this->stats[i][j][1];
            double exponent = exp(-(pow((v[j]-mu),2))/(2*pow(stdev,2)));
            double p = (1/(sqrt(2*M_PI)*stdev)*exponent);
            probabilities[i] *= p;
        }
    }

    //int ans = distance(begin(probabilities), max_element(begin(probabilities), end(probabilities)));

    // use just one for loop
    int i = 0, ans = 0;
    double biggest = 0.0;

    for (auto v: probabilities){
        if(v > biggest){
            biggest = v;
            ans = i;
        }
        i++;
    }
    return this->possible_labels[ans];

}

