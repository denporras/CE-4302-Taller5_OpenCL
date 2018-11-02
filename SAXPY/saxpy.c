#include <stdio.h>
#include <stdlib.h>
// Include OpenCL headers 
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
// What's up with this? :O
#define MAX_SOURCE_SIZE (0x100000)

#include <time.h>
 
int main(void) {
    // Create the two input vectors
    int i;
    const int LIST_SIZE = 90000000;
    printf("Size of vectors: %d\n", LIST_SIZE);
    int *A = (int*)malloc(sizeof(int));
    *A = 10;

    int *X = (int*)malloc(sizeof(int)*LIST_SIZE);

    int *Y = (int*)malloc(sizeof(int)*LIST_SIZE);
    // Init the vectors 
    for(i = 0; i < LIST_SIZE; i++) {
        X[i] = LIST_SIZE - i;
        Y[i] = i;
    }
 
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    // Read the kernel
    fp = fopen("saxpy.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    //1. Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;   
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
            &device_id, &ret_num_devices);
 
    //2. Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
 
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            sizeof(int), NULL, &ret);
    cl_mem x_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem y_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
            LIST_SIZE * sizeof(int), NULL, &ret);
 
    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            sizeof(int), A, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, x_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(int), X, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, y_mem_obj, CL_TRUE, 0, 
            LIST_SIZE * sizeof(int), Y, 0, NULL, NULL);
 
    //3. Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    //4. Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&x_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&y_mem_obj);
 
    //5. Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64

    //Time mesurement variables
    clock_t start , end;
    float run_time;

    start = clock(); //Initial time

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, NULL);

    ret = clEnqueueReadBuffer(command_queue, y_mem_obj, CL_TRUE, 0, 
        LIST_SIZE * sizeof(int), Y, 0, NULL, NULL);
    
    //Run time calc
    end = clock();
    run_time = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time using OpenCL: %f s\n", run_time);

    
    //Do operation with CPU
    start = clock(); //Initial time
    for (int i = 0; i < LIST_SIZE; ++i)
        Y[i] = (*A)*X[i] + Y[i];
    
    end = clock();
    run_time = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time using CPU: %f s\n", run_time);
 
    // Display the result to the screen
 
    //6. Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(x_mem_obj);
    ret = clReleaseMemObject(y_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(X);
    free(Y);
    return 0;
}
