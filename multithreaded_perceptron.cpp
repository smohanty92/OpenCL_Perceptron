//
//  main.cpp
//  threading
//
//  Created by saroj mohanty on 4/30/18.
//  Copyright © 2018 saroj. All rights reserved.
//

#include <iostream>
#include <thread>

//For benchmarking
#include <ctime>

//Number of threads to be used
static const int NUM_THREADS = 9;

//Pre-allocated set of offsets (generated from logging output of sequential algorithm)
static const float b[NUM_THREADS] = {
    50.763,
    50.763,
    50.763,
    25.3815,
    25.3815,
    25.3815,
    25.3815,
    25.3815,
    0
};

//Pre-allocated set of weights
static const float weights [NUM_THREADS][4] = {
    {0.98, 3.08, 2.24, 4.13},
    {0.41, 3.17, 1.25, 3.77},
    {-0.61, 3.2, -0.67, 3.02},
    {0.29, 4.19, -1.24, 2.72},
    {-0.28, 4.28, -2.23, 2.36},
    {-0.85, 4.37, -3.22, 2},
    {-1.42, 4.46, -4.21, 1.64},
    {-1.99, 4.55, -5.2, 1.28},
    {-0.46, 5.6, -4.78, 1.34}
};

//Initialize our training set (First 100 rows of famous Iris dataset. First 4 columns are attributes and last column is label)
static const float trainingSet[100][5] = {
    {5.1,3.5,1.4,0.2,1},
    {4.9,3.0,1.4,0.2,1},
    {4.7,3.2,1.3,0.2,1},
    {4.6,3.1,1.5,0.2,1},
    {5.0,3.6,1.4,0.2,1},
    {5.4,3.9,1.7,0.4,1},
    {4.6,3.4,1.4,0.3,1},
    {5.0,3.4,1.5,0.2,1},
    {4.4,2.9,1.4,0.2,1},
    {4.9,3.1,1.5,0.1,1},
    {5.4,3.7,1.5,0.2,1},
    {4.8,3.4,1.6,0.2,1},
    {4.8,3.0,1.4,0.1,1},
    {4.3,3.0,1.1,0.1,1},
    {5.8,4.0,1.2,0.2,1},
    {5.7,4.4,1.5,0.4,1},
    {5.4,3.9,1.3,0.4,1},
    {5.1,3.5,1.4,0.3,1},
    {5.7,3.8,1.7,0.3,1},
    {5.1,3.8,1.5,0.3,1},
    {5.4,3.4,1.7,0.2,1},
    {5.1,3.7,1.5,0.4,1},
    {4.6,3.6,1.0,0.2,1},
    {5.1,3.3,1.7,0.5,1},
    {4.8,3.4,1.9,0.2,1},
    {5.0,3.0,1.6,0.2,1},
    {5.0,3.4,1.6,0.4,1},
    {5.2,3.5,1.5,0.2,1},
    {5.2,3.4,1.4,0.2,1},
    {4.7,3.2,1.6,0.2,1},
    {4.8,3.1,1.6,0.2,1},
    {5.4,3.4,1.5,0.4,1},
    {5.2,4.1,1.5,0.1,1},
    {5.5,4.2,1.4,0.2,1},
    {4.9,3.1,1.5,0.2,1},
    {5.0,3.2,1.2,0.2,1},
    {5.5,3.5,1.3,0.2,1},
    {4.9,3.6,1.4,0.1,1},
    {4.4,3.0,1.3,0.2,1},
    {5.1,3.4,1.5,0.2,1},
    {5.0,3.5,1.3,0.3,1},
    {4.5,2.3,1.3,0.3,1},
    {4.4,3.2,1.3,0.2,1},
    {5.0,3.5,1.6,0.6,1},
    {5.1,3.8,1.9,0.4,1},
    {4.8,3.0,1.4,0.3,1},
    {5.1,3.8,1.6,0.2,1},
    {4.6,3.2,1.4,0.2,1},
    {5.3,3.7,1.5,0.2,1},
    {5.0,3.3,1.4,0.2,1},
    {7.0,3.2,4.7,1.4,-1},
    {6.4,3.2,4.5,1.5,-1},
    {6.9,3.1,4.9,1.5,-1},
    {5.5,2.3,4.0,1.3,-1},
    {6.5,2.8,4.6,1.5,-1},
    {5.7,2.8,4.5,1.3,-1},
    {6.3,3.3,4.7,1.6,-1},
    {4.9,2.4,3.3,1.0,-1},
    {6.6,2.9,4.6,1.3,-1},
    {5.2,2.7,3.9,1.4,-1},
    {5.0,2.0,3.5,1.0,-1},
    {5.9,3.0,4.2,1.5,-1},
    {6.0,2.2,4.0,1.0,-1},
    {6.1,2.9,4.7,1.4,-1},
    {5.6,2.9,3.6,1.3,-1},
    {6.7,3.1,4.4,1.4,-1},
    {5.6,3.0,4.5,1.5,-1},
    {5.8,2.7,4.1,1.0,-1},
    {6.2,2.2,4.5,1.5,-1},
    {5.6,2.5,3.9,1.1,-1},
    {5.9,3.2,4.8,1.8,-1},
    {6.1,2.8,4.0,1.3,-1},
    {6.3,2.5,4.9,1.5,-1},
    {6.1,2.8,4.7,1.2,-1},
    {6.4,2.9,4.3,1.3,-1},
    {6.6,3.0,4.4,1.4,-1},
    {6.8,2.8,4.8,1.4,-1},
    {6.7,3.0,5.0,1.7,-1},
    {6.0,2.9,4.5,1.5,-1},
    {5.7,2.6,3.5,1.0,-1},
    {5.5,2.4,3.8,1.1,-1},
    {5.5,2.4,3.7,1.0,-1},
    {5.8,2.7,3.9,1.2,-1},
    {6.0,2.7,5.1,1.6,-1},
    {5.4,3.0,4.5,1.5,-1},
    {6.0,3.4,4.5,1.6,-1},
    {6.7,3.1,4.7,1.5,-1},
    {6.3,2.3,4.4,1.3,-1},
    {5.6,3.0,4.1,1.3,-1},
    {5.5,2.5,4.0,1.3,-1},
    {5.5,2.6,4.4,1.2,-1},
    {6.1,3.0,4.6,1.4,-1},
    {5.8,2.6,4.0,1.2,-1},
    {5.0,2.3,3.3,1.0,-1},
    {5.6,2.7,4.2,1.3,-1},
    {5.7,3.0,4.2,1.2,-1},
    {5.7,2.9,4.2,1.3,-1},
    {6.2,2.9,4.3,1.3,-1},
    {5.1,2.5,3.0,1.1,-1},
    {5.7,2.8,4.1,1.3,-1}
};

//Function to be used for each thread
//It takes as input an index to be used when indexing the weights and offset arrays
void train(int index) {

    int mistake = 0;
    
    //for each training row
    for (int i=0; i<100; i++) {
        
        //Split up input and output data
        float xi[4] = {trainingSet[i][0], trainingSet[i][1], trainingSet[i][2], trainingSet[i][3]};
        float yi = trainingSet[i][4];
        
        //Calculate dot product of weights and input data
        int sum = (weights[index][0] * xi[0]) + (weights[index][1] * xi[1]) + (weights[index][2] * xi[2]) + (weights[index][3] * xi[3]);
        
        //Subtract offset from sum
        int aggregate = sum - b[index];
        
        //sign of aggregate
        int sign = (aggregate > 0) - (aggregate < 0);
        
        if (sign != yi) {
            mistake = 1;
     
            //We can break because we know the pair of weights and offsets associated with this thread do not converge
            break;
        }
}
    
    if (!mistake) {
        //std::cout << "CONVERGENCE!!! The final weights are weights[0]: " << weights[index][0] << " and weights[1]: " << weights[index][1] << " and weights[2] is " << weights[index][2] << " and weights[3] is " << weights[index][3] << " and the final offset is " << b[index] << " \n";
    }
    
}

int main(int argc, const char * argv[]) {
    //Start benchmarking
    long start_s=clock();
    
    //Declare threads
    std::thread threads[NUM_THREADS];
    
    //Launch group of threads
    for (int i=0; i<NUM_THREADS; ++i) {
        //Call the 'train' function
        threads[i] = std::thread(train, i);
    }

    //Join all threads to wait until they're finished
    for (int i=0; i<NUM_THREADS; ++i) {
        threads[i].join();
    }
    
    //Stop benchmark and calculate and log elapsed
    long stop_s=clock();
    std::cout << "time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC) << std::endl;
    
    return 0;
}
