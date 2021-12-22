#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <time.h>

// prints the matrix out using printf for debugging purposes
__device__ void printMatrix(float* a, int a_r, int a_c){
    for (int i = 0; i < a_r; ++i){
        for (int j = 0; j < a_c; ++j){
            printf("%f ", a[i*a_c + j]);
        }
        printf("\n"); 
    }
}

// main gradient descent step kernel function that each thread runs
__global__ void grad_desc_step(int numFeatures, int numSamples, float learningRate, 
    float *sample, float *label, float *sumWeights, float* features, float *weight){
    // printf("in kernel\n");
    // printf("sample\n");
    // printMatrix(sample, numFeatures, 1);
    
    // W^T X  (W^T = out_dim x in_dim = 1 x numFeatures; X = numFeatures x numSamples) = 1 x numSamples
    float y_hat = 0;   
    for (int i = 0; i < numFeatures; i++) {
        y_hat += sample[i] * weight[i];
    }

    y_hat = 1/(1+exp(-y_hat));

    // Compute loss
    float bce_loss = label[0] * log(y_hat) + (1 - label[0]) * log(1 - y_hat);

    // Compute gradient X^T (z - y)
    // const int constNumFeatures = numFeatures;
    float grads[30];
    for (int i = 0; i < numFeatures; i++) {
        grads[i] = sample[i] * (y_hat - label[0]);
    }

    // Compute the backward step
    for (int i = 0; i < numFeatures; i++) {
        sumWeights[i] = weight[i] - learningRate * grads[i];
    }
    // printf("sumWeights\n");
    // printMatrix(sumWeights, numFeatures, 1);
    // printf("weight\n");
    // printMatrix(weight, numFeatures, 1);
}

int main(int argc, char *argv[]){
    // creating cuda events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // initializing base variables and printing out parameters
    std::cout << "Started program" << std::endl;
    int numSamples = 256000;
    int numFeatures = 30;
    float learningRate = .1;
    int numThreads = 128;
    std::cout << "numSamples: " << numSamples << std::endl;
    std::cout << "numFeatures: " << numFeatures << std::endl;
    std::cout << "learningRate: " << learningRate << std::endl;
    std::cout << "numThreads: " << numThreads << std::endl;

    // creates features and sends it over to the kernel
    float features[numSamples][numFeatures], *featuresDevice;
    for (int i = 0; i < numSamples*numFeatures; i++){
        features[i/numFeatures][i%numFeatures] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;//1+i;
    }
    const size_t featuresSize = sizeof(float) * size_t(numSamples*numFeatures);
    cudaMalloc((void **)&featuresDevice, featuresSize);
    cudaMemcpy(featuresDevice, features, featuresSize, cudaMemcpyHostToDevice);


    // creates labels and sends it over to the kernel
    float labels[numSamples][1], *labelsDevice;
    for (int i = 0; i < numSamples*1; i++){
        labels[i/1][i%1] = 1+i;
    }
    const size_t labelsSize = sizeof(float) * size_t(numSamples*1);
    cudaMalloc((void **)&labelsDevice, labelsSize);
    cudaMemcpy(labelsDevice, labels, labelsSize, cudaMemcpyHostToDevice);


    // initialize _weight vector, where -1 <= w_i <= 1 and sends it over to kernel
    float weight[numFeatures][1], *weightDevice;
    for (int i = 0; i < numFeatures; i++){
        weight[i][0] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
    }
    const size_t weightSize = sizeof(float) * size_t(numFeatures*1);
    cudaMalloc((void **)&weightDevice, weightSize);
    cudaMemcpy(weightDevice, weight, weightSize, cudaMemcpyHostToDevice);


    // training section
    // breaking down numSamples into batches of numThreads
    cudaEventRecord(start);
    int sampleNum = 0;
    for (int i = 0; i < numSamples/numThreads; ++i){
        // std::cout << "iteration: " << i << std::endl;

        // initialize the weight sum array and send it over the kernel
        std::vector<float*> sums(numThreads);
        for (int j = 0; j < numThreads; j++){
            float weightSum[numFeatures][1], *weightSumDevice;
            for (int k = 0; k < numFeatures; k++){
                weightSum[k][0] = 0;
            }
            const size_t weightSumSize = sizeof(float) * size_t(numFeatures*1);
            cudaMalloc((void **)&weightSumDevice, weightSumSize);
            cudaMemcpy(weightSumDevice, weightSum, weightSumSize, cudaMemcpyHostToDevice);
            sums[j] = weightSumDevice;
        }

        // run each batch of numThreads num samples at once and pass in respective sum vectors
        // once they are all done and synchronized, update the regular weight vector with the average
        std::vector<float*> samplePointers(numThreads);
        std::vector<float*> labelPointers(numThreads);
        for (int j = 0; j < numThreads; j++){
            // std::cout << "In second for loop, j= " << j << std::endl;
        
            // creating the sample to operate on
            float sample[numFeatures][1], *sampleDevice;
            for (int k = 0; k < numFeatures; k++){
                sample[k][0] = features[sampleNum][k];
            }
            ++sampleNum;
            const size_t sampleSize = sizeof(float) * size_t(numFeatures*1);
            cudaMalloc((void **)&sampleDevice, sampleSize);
            cudaMemcpy(sampleDevice, sample, sampleSize, cudaMemcpyHostToDevice);
            samplePointers[j] = sampleDevice;

            // creating the label to operate on
            float label[1][1], *labelDevice;
            label[0][0] = labels[sampleNum][0];
            const size_t labelSize = sizeof(float) * size_t(1*1);
            cudaMalloc((void **)&labelDevice, labelSize);
            cudaMemcpy(labelDevice, label, labelSize, cudaMemcpyHostToDevice);
            labelPointers[j] = labelDevice;

            // std::cout << "Made and transferred sample and label" << std::endl;
            grad_desc_step<<<1, 1>>>(numFeatures, numSamples, learningRate, sampleDevice, labelDevice, sums[j], featuresDevice, weightDevice);
        }
        // waits for the numThreads to finish
        cudaDeviceSynchronize();

        // freeing up the previous samples and labels to run again
        for (int j = 0; j < numThreads; j++){
            cudaFree(samplePointers[j]);
            cudaFree(labelPointers[j]); 
        }
        
        // to sum up the weight sums to average them later
        float weightSumFull[numFeatures][1];
        for (int j = 0; j < numFeatures; j++){
            weightSumFull[j][0] = 0;
        }
        for (int j = 0; j < numThreads; j++){
            float weightSumCurr[numFeatures][1];
            cudaMemcpy(weightSumCurr, sums[j], weightSize, cudaMemcpyDeviceToHost);
            cudaFree(sums[j]);
            for (int k = 0; k < numFeatures; k++){
                weightSumFull[k][0] += weightSumCurr[k][0];
            }
        }
        for (int j = 0; j < numFeatures; j++){
            weight[j][0] = weightSumFull[j][0] / numThreads;
        }
        // frees up the old weight data on device and recopies new weight data to device for next iteration
        cudaFree(weightDevice);
        cudaMalloc((void **)&weightDevice, weightSize);
        cudaMemcpy(weightDevice, weight, weightSize, cudaMemcpyHostToDevice);
        // printf("\naveraged out weight for batch: %d  \n", i);
        // for (int j = 0; j < numFeatures; j++){
        //     printf("%f ", weight[j][0]);
        // }
    }
    // stops the timer and frees up the remaining cuda data
    cudaEventRecord(stop);
    cudaFree(featuresDevice);
    cudaFree(labelsDevice);
    cudaFree(weightDevice);

    // prints out the time it took to run the program
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %.5fs\n", (double)(milliseconds/1000));
    std::cout << "finished" << std::endl;
}