//
// Created by Tianzi Harrison on 7/30/17.
//

#ifndef T3L3_NAIVEBAYESC_CLASSIFIER_H
#define T3L3_NAIVEBAYESC_CLASSIFIER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {
public:

    vector<string> possible_labels = {"left", "keep", "right"};

    vector<vector<vector<double>>> stats;

    /**
      * Constructor
      */
    GNB();

    /**
     * Destructor
     */
    virtual ~GNB();

    void train(vector<vector<double> > data, vector<string> labels);

    string predict(vector<double> v);

    double find_mean(vector<double> v);
    double find_stdev(vector<double> v);

};

#endif //T3L3_NAIVEBAYESC_CLASSIFIER_H




