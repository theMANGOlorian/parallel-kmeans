/*
 * k-Means clustering algorithm
 *
 * OpenMP version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <omp.h>

#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* 
Function showFileError: It displays the corresponding error during file reading.
*/
void showFileError(int error, char* filename)
{
	printf("Error\n");
	switch (error)
	{
		case -1:
			fprintf(stderr,"\tFile %s has too many columns.\n", filename);
			fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
			break;
		case -2:
			fprintf(stderr,"Error reading file: %s.\n", filename);
			break;
		case -3:
			fprintf(stderr,"Error writing file: %s.\n", filename);
			break;
	}
	fflush(stderr);	
}

/* 
Function readInput: It reads the file to determine the number of rows and columns.
*/
int readInput(char* filename, int *lines, int *samples)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;
    
    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL) 
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1;
			}
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL)
            {
            	contsamples++;
				ptr = strtok(NULL, delim);
	    	}	    
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;  
        return 0;
    }
    else
	{
    	return -2;
	}
}

/* 
Function readInput2: It loads data from file.
*/
int readInput2(char* filename, float* data)
{
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;
    
    if ((fp=fopen(filename,"rt"))!=NULL)
    {
        while(fgets(line, MAXLINE, fp)!= NULL)
        {         
            ptr = strtok(line, delim);
            while(ptr != NULL)
            {
            	data[i] = atof(ptr);
            	i++;
				ptr = strtok(NULL, delim);
	   		}
	    }
        fclose(fp);
        return 0;
    }
    else
	{
    	return -2; //No file found
	}
}

/* 
Function writeResult: It writes in the output file the cluster of each sample (point).
*/
int writeResult(int *classMap, int lines, const char* filename)
{	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL)
    {
        for(int i=0; i<lines; i++)
        {
        	fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
   
        return 0;
    }
    else
	{
    	return -3; //No file found
	}
}


/*

Function initCentroids: This function copies the values of the initial centroids, using their 
position in the input data structure as a reference map.
*/
void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K)
{
	int i;
	int idx;
	for(i=0; i<K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
	}
}

/*
Function euclideanDistance: Euclidean distance
This function could be modified
*/
float euclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		dist+= (point[i]-center[i])*(point[i]-center[i]);
	}
	dist = sqrt(dist);
	return(dist);
}

/**
Annotazioni: EuclideanDistance2
    Rimossa la radice quadrata per evitare calcoli inutili.
	Ho provato ad usare #pragma omp simd reduction(+:dist) ma non funziona perche il float non è associativo
	Ho provato ad usare fmaf + simd ma non funziona perche c'è una dipendenza tra le iterazioni (ovvero dist_squared)
	Ho provato a creare un array temporaneo per memorizzare i risultati parziali e poi sommarli tutti insieme per usare simd
 */

float euclideanDistance2(float *point1, float *center, int samples) {
    float dist_squared = 0.0;
    for (int i = 0; i < samples; i++) {
        float diff = point1[i] - center[i];
       	dist_squared += diff * diff;
    }
    return dist_squared; // Restituisce la somma dei quadrati delle differenze
}


/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	#pragma omp for private(i,j) schedule(static)
	for (int k=0; k<rows*columns; k++){
        i = k/columns;
		j = k - i*columns; //j = k%columns; operation is equivalent to the previous one
        matrix[i*columns+j] = 0.0;
    }
	
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	#pragma omp for schedule(static)
	for (int i=0; i<size; i++)
		array[i] = 0;	
}

void parallelMemcpy(float* A, float* B, int size)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < size; i++)
    {
		A[i] = B[i];
    }
}

void zeroIntShared(int *array, int size)
{
	for (int i=0; i<size; i++)
		array[i] = 0;	
}
void zeroFloatShared(float *array, int rows, int columns)
{
	for (int i=0; i<rows; i++)
		for (int j=0; j<columns; j++)
			array[i*columns+j] = 0.0;	
}


int main(int argc, char* argv[])
{
	printf("\nKmeans omp v.2\n");
	int argv7 = atoi(argv[7]);
	omp_set_num_threads(argv7);
	int numberThreads = omp_get_max_threads();
    printf("\nNumbers of threads: %d\n", numberThreads);

	//START CLOCK***************************************
	double start, end;
	double totalTime = 0.0;
	start = omp_get_wtime();
	//**************************************************
	/*
	* PARAMETERS
	*
	* argv[1]: Input data file
	* argv[2]: Number of clusters
	* argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	* argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	*          If between one iteration and the next, the percentage of class changes is less than
	*          this percentage, the algorithm stops.
	* argv[5]: Precision in the centroid distance after the update.
	*          It is an algorithm termination condition. If between one iteration of the algorithm 
	*          and the next, the maximum distance between centroids is less than this precision, the
	*          algorithm stops.
	* argv[6]: Output file. Class assigned to each point of the input file.
	* */
	if(argc !=  8)
	{
		fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
	int lines = 0, samples= 0;  
	
	int error = readInput(argv[1], &lines, &samples);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}
	
	float *data = (float*)calloc(lines*samples,sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if(error != 0)
	{
		showFileError(error,argv[1]);
		exit(error);
	}

	// Parameters
	int K=atoi(argv[2]); 
	int maxIterations=atoi(argv[3]);
	int minChanges= (int)(lines*atof(argv[4])/100.0);
	float maxThreshold=atof(argv[5]);

	int *centroidPos = (int*)calloc(K,sizeof(int));
	float *centroids = (float*)calloc(K*samples,sizeof(float));
	int *classMap = (int*)calloc(lines,sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(0);
	int i;
	for(i=0; i<K; i++) 
		centroidPos[i]=rand()%lines;
	
	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);


	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);
	
	//END CLOCK*****************************************
	end = omp_get_wtime();

	printf("\nMemory allocation: %f seconds\n", end - start);
	totalTime += end - start;
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it=0;
	int changes = 0;
	float maxDist;

	//pointPerClass: number of points classified in each class
	//auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K*sizeof(int));
	float *auxCentroids = (float*)malloc(K*samples*sizeof(float));
	float *distCentroids = (float*)malloc(K*sizeof(float)); 
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr,"Memory allocation error.\n");
		exit(-4);
	}

	
/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */	

	int threadID;
	// array condivisi tra i threads per memorizzare il numero di punti per ogni classe
	int **shared_vectors = malloc(numberThreads * sizeof(int *));
	#pragma omp parallel for
	for (int x = 0; x < numberThreads; x++) {
		shared_vectors[x] = malloc(K * sizeof(int));
	}
	float **shared_centroids = malloc(numberThreads * sizeof(float *));
	#pragma omp parallel for
	for (int x = 0; x < numberThreads; x++) {
		shared_centroids[x] = malloc(K * samples * sizeof(float));
	}

	// creo la regione parallela
	#pragma omp parallel shared(it, changes, maxDist, outputMsg, shared_vectors, shared_centroids) private(threadID)
	{
		do{
			#pragma omp barrier
			#pragma omp single
			{
				it++;
				changes = 0;
			}
			//1. Calculate the distance from each point to the centroid
			//Assign each point to the nearest centroid.
			#pragma omp for private(i,j,class,dist,minDist) reduction(+:changes) schedule(dynamic) // attenzione, se il calcolo dei chunk = 0 allora crea un loop
			for(i=0; i<lines; i++)
			{
				class=1;
				minDist=FLT_MAX;
				for(j=0; j<K; j++)
				{
					// rimuovere la chiamata alla funzione, (potrebbe generare overhead di chiamata)
					//dist=euclideanDistance(&data[i*samples], &centroids[j*samples], samples);

					float tempDist=0.0;
					float *point = &data[i*samples];
					float *center = &centroids[j*samples];
					for(int d=0; d<samples; d++) 
					{
						tempDist+= (point[d]-center[d])*(point[d]-center[d]);
					}
					dist = sqrtf(tempDist);

					if(dist < minDist)
					{
						minDist=dist;
						class=j+1;
					}
				}
				if(classMap[i]!=class)
				{
					changes++;
				}
				classMap[i]=class;
			}
			threadID = omp_get_thread_num();
			// 2. Recalculates the centroids: calculates the mean within each cluster

			#pragma omp for private(i) schedule(static)
			for (int i = 0; i < numberThreads; i++) {
				zeroIntShared(shared_vectors[i], K);
			}
			#pragma omp for private(i) schedule(static)
			for (int i = 0; i < numberThreads; i++) {
				zeroFloatShared(shared_centroids[i], K, samples);
			}

			zeroIntArray(pointsPerClass,K);
			zeroFloatMatriz(auxCentroids,K,samples);

			#pragma omp for private(i,j,class)
			for(i=0; i<lines; i++) 
			{
				class=classMap[i];
				shared_vectors[threadID][class-1]++;
				for(j=0; j<samples; j++){
					shared_centroids[threadID][(class-1)*samples+j] += data[i*samples+j];
				}
			}
			//  somma le colonne di shared_vectors parallelamente, cosi non race condition tra i threads
			#pragma omp for schedule(static)
			for (int col = 0; col < K;col++) {
				for (int i = 0; i < numberThreads; i++) {
					pointsPerClass[col] += shared_vectors[i][col];
				}	
			}
			#pragma omp for schedule(static)
			for (int col = 0; col < K * samples; col++) {
				for (int i = 0; i < numberThreads; i++) {
					auxCentroids[col] += shared_centroids[i][col];
				}
			}

			#pragma omp for private(i,j) schedule(static)
			for (int t = 0; t < K*samples; t++)
			{
				i = t / samples;
				j = t - i * samples; // j = t % samples; operation is equivalent to the previous one
				auxCentroids[i*samples+j] /= pointsPerClass[i];
			}
			
			#pragma omp single
			{
				maxDist=FLT_MIN;
			}
			#pragma omp for reduction(max:maxDist)
			for(i=0; i<K; i++){
				distCentroids[i]=euclideanDistance(&centroids[i*samples], &auxCentroids[i*samples], samples);
				if(distCentroids[i]>maxDist) {
					maxDist=distCentroids[i];
				}
			}
			parallelMemcpy(centroids, auxCentroids, K*samples); //memcpy(centroids, auxCentroids, (K*samples*sizeof(float)));
				
			#pragma omp single 
			{
				sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
				outputMsg = strcat(outputMsg,line);
			}
			
		} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));
	}
    

/*  
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\nComputation: %f seconds", end - start);
	fflush(stdout);
	totalTime += end - start;
	//**************************************************
	//START CLOCK***************************************
	start = omp_get_wtime();
	//**************************************************

	if (changes <= minChanges) {
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations) {
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else {
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}	

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if(error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	//Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	//free shared memory
	// Liberare memory per shared_vectors
	for (int x = 0; x < numberThreads; x++) {
		free(shared_vectors[x]);  // Libera ogni array di interi
		free(shared_centroids[x]);  // Libera ogni array di float
	}
	free(shared_vectors);  // Libera l'array di puntatori
	free(shared_centroids);  // Libera l'array di puntatori


	//END CLOCK*****************************************
	end = omp_get_wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end - start);
	fflush(stdout);
	//***************************************************/
	return 0;
}
/*
Crea N array condivisi tra i threads
Dopo i calcoli, ogni thread somma ogni indice di questi array condivisi
Non c'è bisogno di usare atomic o crtical o locking
*/
