//Include OpenCL Framework
#include <OpenCL/opencl.h>

//For benchmarking
#include "time.h"

//Number of work items
#define NUM_WORK_ITEMS (9)

//Kernel function
//Kernels in OpenCL are just text that get dynamically compiled by the host code
const char *KernelSource = "\n" \
"__kernel void train(                                                       \n" \
//Pass in global float pointers of the training set, weights, and offsets (b)
" __global float* trainingSet, __global float* weights, __global float* b) \n" \

"{                                                                      \n" \

//Get the global id of the current work item (To be used for indexing the weights and offsets array
"   int i = get_global_id(0);                                           \n" \

//Obtain all 4 weights with respect to the current index
"   float w1 = weights[i*4];                                                     \n" \
"   float w2 = weights[i*4+1];                                                    \n" \
"   float w3 = weights[i*4+2];                                                    \n" \
"   float w4 = weights[i*4+3];                                                    \n" \

//Keep track of if a mistake was made in classifying
"   int mistake = 0;                                                     \n" \

//Loop through every row in the training set
//We iterate through 500 instead of 100 (and skip every 5) because 2d arrays have to be recognized as a 1-dimensional contiguous block of memory
"   for (int j=0; j<500; j=j+5) {                                          \n" \

//Obtain all input data of the current row
"       float x1 = trainingSet[j];                                        \n" \
"       float x2 = trainingSet[j+1];                                      \n" \
"       float x3 = trainingSet[j+2];                                      \n" \
"       float x4 = trainingSet[j+3];                                      \n" \

//Obtain label of row
"       float yi = trainingSet[j+4];                                       \n" \

//Compute dot product of weights and input vector. Then subtract from offset
"       float aggregate = ((w1*x1) + (w2*x2) + (w3*x3) + (w4*x4)) - b[i];                \n" \

//Get the sign of the aggregate
"       int sign = (aggregate > 0) - (aggregate < 0);                                  \n" \

//Set mistake=1 if there was an error. Then break because it's meaningless to continue as we know this pair of weights and offsets do not converge
"       if (sign != yi) {                                                               \n" \
"           mistake = 1;                                                        \n" \
"           break;                                                               \n" \
"       }                                                                       \n" \
"   }                                                                   \n" \

//If no mistake was made, the algorithm has converged successfully
"   if (!mistake) {                                                     \n" \
"       //printf(\"CONVERGENCE! The index for the weights and offset that caused the convergence is %d \\n\", i);                                  \n" \
"   }                                                                   \n" \

"}                                                                      \n" \
"\n";

//Main Host code
int main(int argc, char** argv)
{
    //Begin benchmarking
    clock_t begin = clock();

    //Pre-allocated array of offsets
    float b[9] = {
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
    
    //Pre-allocated 2d array of weights (weights have same length as input data)
    float weights [9][4] = {
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
  
    //In OpenCL you have to always check for error flags or else the Kernel code can cause the Host or OS to crash
    int err;
    
    //Global and local domain sizes (used when executing kernel)
    size_t global;
    size_t local;

    //The actual device being used
    cl_device_id device_id;
    
    //The context
    cl_context context;
    
    //The command queue being used
    cl_command_queue commands;
    
    //The program that will be compiled
    cl_program program;
    
    //The kernel object
    cl_kernel kernel;
 
    //Initialize our training set (This is a truncated version of the famous Iris dataset that is linearly separable. It contains 4 attributes and 1 label)
    float trainingSet[100][5] = {
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

    //Flag for whether we want to use GPU or CPU
    int gpu = 1;
    
    //Connect to device
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: COULD NOT CONNECT TO DEVICE \n");
        return EXIT_FAILURE;
    }
  
    //Create a context with the device
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("ERROR: COULD NOT CREATE CONTEXT \n");
        return EXIT_FAILURE;
    }

    //Create the command queue with the context and device
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("ERROR: COULD NOT CREATE QUEUE \n");
        return EXIT_FAILURE;
    }

    //Create the program from the kernel source code above
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program) {
        printf("ERROR: COULD NOT CREATE THE PROGRAM \n");
        return EXIT_FAILURE;
    }

    //Try to compile the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
     
        //If the program failed to compile, allocate a buffer to read in the error code
        size_t len;
        char buffer[2048];

        printf("ERROR: COULD NOT COMPILE THE PROGRAM. ERROR CODE IS: \n");
    
        //Get the error code from the program build log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        
        //Log error and exit
        printf("%s\n", buffer);
        exit(1);
    }

    //Create the kernel for the program. Call the 'train' function of the kernel source code
    kernel = clCreateKernel(program, "train", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("ERROR: COULD NOT CREATE KERNEL! \n");
        exit(1);
    }
    
    //Output memory buffers for training set and free parameters
    cl_mem trainingSetBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(trainingSet), NULL, NULL);
    cl_mem weightsBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(weights), NULL, NULL);
    cl_mem offsetBuf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(b), NULL, NULL);
    
    if (!trainingSetBuf || !weightsBuf || !offsetBuf) {
        printf("ERROR: COULD NOT ALLOCATE MEMORY FOR PARAMETERS \n");
    }
    
    //Enqueue write buffer for training set
    err = clEnqueueWriteBuffer(commands, trainingSetBuf, CL_TRUE, 0, sizeof(trainingSet), &trainingSet, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: COULD NOT WRITE TO trainingSetBuf! \n");
        exit(1);
    }
    
    //Enqueue write buffer for weights
    err = clEnqueueWriteBuffer(commands, weightsBuf, CL_TRUE, 0, sizeof(weights), &weights, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: COULD NOT WRITE TO weightsBuf! \n");
        exit(1);
    }
    
    //Enqueue write buffer for offsets
    err = clEnqueueWriteBuffer(commands, offsetBuf, CL_TRUE, 0, sizeof(b), &b, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: COULD NOT WRITE TO offsetBuf! \n");
        exit(1);
    }
    
    //Set arguments for kernel parameters (in order)
    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &trainingSetBuf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &weightsBuf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &offsetBuf);
    
    if (err != CL_SUCCESS) {
        printf("ERROR: COULD NOT SET KERNEL PARAMETERS! ERROR CODE IS: %d\n", err);
        exit(1);
    }

    //Obtain the maximum work group size for executing the kernel on this device
    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("ERROR: FAILED TO RETRIEVE WORK GROUP INFO FOR THIS DEVICE! ERROR CODE IS:  %d\n", err);
        exit(1);
    }

    //Dispatch the kernel over the range of our dataset (data-parallel programming)
    global = NUM_WORK_ITEMS;

    //Finally execute the kernel
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    if (err) {
        printf("ERROR: COULD NOT EXECUTE KERNEL! ERROR CODE IS: %d\n", err);
        return EXIT_FAILURE;
    }

    //Wait for everything in the command queue to finish
    clFinish(commands);
    
    //Release all memory buffers, the program, kernel, queue, and context
    clReleaseMemObject(trainingSetBuf);
    clReleaseMemObject(weightsBuf);
    clReleaseMemObject(offsetBuf);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    //end benchamrking
    clock_t end = clock();

    //calculate elapsed time
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    //log benchmarking
    printf("execution time spent was %f\n", time_spent);
    
    return 0;
}

