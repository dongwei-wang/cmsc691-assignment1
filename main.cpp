#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <mutex>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

//mutex m;

int* KNN(ArffData* dataset){
    int* predictions = (int*)malloc(dataset->num_instances() * sizeof(int));
    for(int i = 0; i < dataset->num_instances(); i++){
        float smallestDistance = FLT_MAX;
        int smallestDistanceClass;
		// target each other instance
        for(int j = 0; j < dataset->num_instances(); j++){
            if(i == j)
				continue;
            float distance = 0;
			// compute the distance between the two instances
            for(int k = 0; k < dataset->num_attributes() - 1; k++)             {
                float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
                distance += diff * diff;
            }
            distance = sqrt(distance);
			// select the closest one
            if(distance < smallestDistance){
                smallestDistance = distance;
                smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes()-1)->operator int32();
            }
        }
        predictions[i] = smallestDistanceClass;
    }
    return predictions;
}

// This is the thread function for parallel
void* KNN_thread(ArffData* dataset, int *predictions, int start, int gap){
	for( int i = start; i<start+gap; i++ ){
		float smallestDistance = FLT_MAX;
		int smallestDistanceClass;

		for(int j = 0; j < dataset->num_instances(); j++){
			if(j == i)
				continue;

			float distance = 0;
			// compute the distance between the two instances
			for(int k = 0; k < dataset->num_attributes()-1; k++){
				float diff = dataset->get_instance(i)->get(k)->operator float() - dataset->get_instance(j)->get(k)->operator float();
				distance += diff * diff;
			}
			distance = sqrt(distance);
			// select the closest one
			if(distance < smallestDistance){
				smallestDistance = distance;
				smallestDistanceClass = dataset->get_instance(j)->get(dataset->num_attributes()-1)->operator int32();
			}
		}
		predictions[i] = smallestDistanceClass;
	}
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int));
	// for each instance compare the true class and predicted class
    for(int i = 0; i < dataset->num_instances(); i++){
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes()-1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;
    for(int i = 0; i < dataset->num_classes(); i++){
		// elements in the diagnoal are correct predictions
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i];
    }
    return successfulPredictions / (float) dataset->num_instances();
}

void maitrixdisplay(int* matrix, int size){
	for(int i=0; i<size; i++){
		for(int j=0;j<size;j++){
			printf("%4d", matrix[i*size+j]);
		}
		printf("\n");
	}
}

bool matrixcompare(int* matrix1, int* matrix2, int size){
	for(int i=0; i<size*size; i++){
		if(matrix1[i] - matrix2[i] != 0)
			return false;
	}
	return true;
}


int main(int argc, char *argv[])
{
    if(argc != 2){
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }

    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
	printf("The number of instances is %ld\n", dataset->num_instances());

	struct timespec start, end;
	uint64_t diff;
	printf("\n***** sequential processing *****\n");
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    int* predictions = KNN(dataset);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accu_sequential = computeAccuracy(confusionMatrix, dataset);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("The 1NN classifier in a SEQUENTIAL manner for %lu instances required %llu ms CPU time, accuracy was %.4f\n",
			dataset->num_instances(), (long long unsigned int)diff, accu_sequential);

	printf("\n***** multi thread processing *****\n");
	int thread_cnt;
	printf("Please input the number of threads!\n");
	scanf("%d", &thread_cnt);

	if(thread_cnt <=0||thread_cnt > dataset->num_instances()){
		printf("Thread number should be greater than 0 and no greater than the number of instance!\n");
		exit(1);
	}

	int* prediction_mt = (int*)malloc(dataset->num_instances()*sizeof(int));

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	thread *threads = new thread[thread_cnt];
	int quotient = dataset->num_instances()/thread_cnt;
	int reminder = dataset->num_instances()%thread_cnt;
	int gap;
	int begin = 0;
	// launch threads for parallel execution
	for( int i=0; i<thread_cnt; i++){
		gap = i<reminder ? quotient+1 : quotient;
		threads[i] = thread(KNN_thread, dataset, prediction_mt, begin, gap);
		begin += gap;
	}

	for( int i=0; i<thread_cnt; i++ ){
		threads[i].join();
	}

	int *cm_mt = computeConfusionMatrix(prediction_mt,dataset);
    float accr_mt = computeAccuracy(cm_mt, dataset);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;
    printf("The 1NN classifier in a PARALLEL manner for %lu instances required %llu ms CPU time, accuracy was %.4f\n",
			dataset->num_instances(), (long long unsigned int)diff, accr_mt);

	// printf("\n******** SEQUENTIAL prediction matrix ********\n");
	// maitrixdisplay(confusionMatrix, dataset->num_classes());

	// printf("\n******** PARALLEL prediction matrix ********\n");
	// maitrixdisplay(cm_mt, dataset->num_classes());

	printf("Compare the two confusion matrix of sequential and parallel......\n");
	if( matrixcompare(cm_mt, confusionMatrix, dataset->num_classes()) )
		printf("Test Passed!\n");
	else
		printf("Test Failed!\n");


}
