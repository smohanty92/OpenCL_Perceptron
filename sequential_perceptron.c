#include <stdio.h>
#include <math.h>

//For benchmarking
#include <time.h>

int main(int argc, const char * argv[]) {
   
    //Start benchmarking
    float startTime = (float)clock()/CLOCKS_PER_SEC;
    
    //Initialize our training set (This comes from the first 100 rows of the famous Iris dataset)
    //The first four columns are the input attributes and the last column in the label
    // 1 & -1 denote two different species of flowers that can be linearly separable
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
    
    //Row and column sizes of trainingset
    static const int ROW_SIZE = 100;
    static const int COLUMN_SIZE = 4; //excludes y
    
    //Initialize weights to 5 and of same length of inputs
    float weights[4] = {5.0,5.0,5.0,5.0};
    
    //Initialize offset term to 0
    float b = 0;
    
    //Learning rate
    float n = 0.3;
    
    float radius = 0;
    //Calculate the radius by obtaining the max length of all input data rows
    for (int i=0; i<ROW_SIZE; i++) {
        //Loop through rows
        
        float sum = 0;
        
        //Loop through columns
        for (int j=0; j<COLUMN_SIZE; j++) {
            sum += (trainingSet[i][j] * trainingSet[i][j]);
        }
        
        float length = sqrt(sum);
        
        if (length > radius) {
            radius = length;
        }
    }
    
    //Until the search converges and stops making mistakes
    while (1) {
        int mistake = 0;
        
        //Loop through all rows
        for (int i = 0; i<ROW_SIZE; i++) {
            
            int xi[4] = {trainingSet[i][0], trainingSet[i][1], trainingSet[i][2], trainingSet[i][3]};
            int yi = trainingSet[i][4];
            
            //compute dot product of weights and input data (W*Xi)
            int sum = 0;
            
            for (int j=0; j<COLUMN_SIZE; j++) {
                sum += weights[j] * xi[j];
            }
            
            //Result of aggregation function plus offset term
            int aggregation = sum - b;
            
            //sign of aggregate
            int sign = (aggregation > 0) - (aggregation < 0);
            
            //Check if result of aggregate function does not match label point
            if (sign != yi) {
                //mistake was made
                mistake = 1;
                
                //Update weights
                for (int j=0; j<COLUMN_SIZE; j++) {
                    weights[j] = weights[j] + n*yi*xi[j];
                }
                
                //Update offset
                b = b - n*yi*radius*radius;
            }
            
        }
        
        if (!mistake) {
            break;
        }
        
    }
    
    //Calculate and log elapsed time
    float endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = endTime - startTime;
    printf("timeElapsed is %f \n", timeElapsed);

    //Print the resulting weights and offset free parameters that caused the convergence
    printf("b is %f\n", b);
    printf("w[0] is %f\n", weights[0]);
    printf("w[1] is %f\n", weights[1]);
    printf("w[2] is %f\n", weights[2]);
    printf("w[3] is %f\n", weights[3]);
    
}
