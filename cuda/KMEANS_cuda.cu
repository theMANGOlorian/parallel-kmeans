/*
 * k-Means clustering algorithm
 *
 * CUDA version
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

		N.B.
		Dear students,
		I have been notified of an issue with the filesystem on node126 on the cluster. 
		You can exclude it from your allocations by adding -append 'requirements = (Machine != "node126.di.rm1")' when running condor_submit
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>


#define MAXLINE 2000
#define MAXCAD 200

//Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL( a )	{ \
	cudaError_t ok = a; \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}
#define CHECK_CUDA_LAST()	{ \
	cudaError_t ok = cudaGetLastError(); \
	if ( ok != cudaSuccess ) \
		fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString( ok ) ); \
	}

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


int writeTime(float t, const char* filename) {

		t /= 1000;

        FILE *fp = fopen(filename, "a");
        if (fp == NULL) {
                fprintf(stderr, "Error opening file: %s\n", filename);
                return 1;
        }

        fprintf(fp, "%f\n", t);
        fclose(fp);
        return 0;
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
		float diff = point[i]-center[i];
		dist+= diff*diff;
	}
	dist = sqrt(dist);
	return(dist);
}

__device__ float CudaEuclideanDistance(float *point, float *center, int samples)
{
	float dist=0.0;
	for(int i=0; i<samples; i++) 
	{
		float diff = point[i]-center[i];
		dist+= diff*diff;
	}
	dist = sqrtf(dist);
	return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i,j;
	for (int k=0; k<rows*columns; k++){
		i = k/columns;
		j = k - i*columns;
		matrix[i*columns+j] = 0.0;
	}
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i=0; i<size; i++)
		array[i] = 0;	
}


// Portare i centroidi nella shared memory per evitare accessi globali lenti | attenzione si puo usurare la shared memory per K * samples troppo grande -> soluzione tiling
// Non possiamo mettere tutto data in shared memory, ma possiamo mettere solo i punti usati in ogni blocco (usare numero di thread troppo grande usura la shared memory)
__global__ void KernelClusterAssignment(float* d_data, float* d_centroids, int* d_classMap, int* d_changes,int d_lines, int d_K, int d_samples, int* d_pointsPerClass, float* d_auxCentroids) {
   
    extern __shared__ unsigned char sharedMemory[]; // uso unsigned char perche è 1 byte, e quindi è piu naturale usare gli indici
    int* sharedChanges = (int*)&sharedMemory[0];
    float* sharedCentroids = (float*)&sharedMemory[sizeof(int)]; // qui si potrebbe fare tiling  per evitare di usare troppa shared memory
    float* localLines = (float*)&sharedMemory[sizeof(int) + sizeof(float) * d_K * d_samples];

    int id = blockIdx.x * blockDim.x + threadIdx.x; // id globale
    int tid = threadIdx.x; // id locale nel blocco

    // Copia centroids in shared memory (solo una volta per blocco)
    for (int i = tid; i < d_K * d_samples; i += blockDim.x) {
        sharedCentroids[i] = d_centroids[i];
    }
    __syncthreads();

    // Copia dati del punto nel blocco corrente (solo se il punto esiste)
    if (id < d_lines) {
        for (int i = 0; i < d_samples; i++) {
            localLines[tid * d_samples + i] = d_data[id * d_samples + i];
        }
    }
    __syncthreads();

    if (tid == 0) {
        *sharedChanges = 0; 
    }
    __syncthreads();

    if (id < d_lines) {
        float minDist = FLT_MAX;
        int cluster = 1;

        for (int j = 0; j < d_K; j++) {
            float dist = CudaEuclideanDistance(&localLines[tid * d_samples], &sharedCentroids[j * d_samples], d_samples);
            if (dist < minDist) {
                minDist = dist;
                cluster = j + 1;
            }
        }

        if (d_classMap[id] != cluster) {
            atomicAdd(sharedChanges, 1);
        }

        d_classMap[id] = cluster;
    }
    __syncthreads();

	// questo prima era in un kernel cuda separato. Ma lo metto qui per sfruttare il fatto che locallines è già in shared memory
    if (id < d_lines) {
        int cluster = d_classMap[id];  // Ottieni la classe del punto i
        atomicAdd(&d_pointsPerClass[cluster - 1], 1);  // Incrementa il numero di punti per la classe (atomic per evitare condizioni di race)

        for (int j = 0; j < d_samples; j++) {
            // Somma i valori dei dati al centroide corrispondente
            atomicAdd(&d_auxCentroids[(cluster - 1) * d_samples + j], localLines[tid * d_samples + j]);
        }
    }

    __syncthreads();
    // somma su memoria globale 
    if (tid == 0) {
        atomicAdd(d_changes, *sharedChanges);
    }
}


__global__ void KenelUpdateCentroids(float* auxCentroids, int* pointsPerClass, int K, int samples){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < K * samples) {
		int i = idx / samples;
		int j = idx - i * samples;
		auxCentroids[i*samples+j] /= pointsPerClass[i];
	}
}

__global__ void KernelMaxDistance(float* centroids, float* auxCentroids, int K, int samples, float *maxDist){
	
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float localMax;
    float dist;
    
    if(tid == 0) {
        localMax = 0;
    }
    __syncthreads();

    if(id < K){
        dist = CudaEuclideanDistance(&auxCentroids[id * samples], &centroids[id * samples], samples);
        atomicMax((int*)&localMax, __float_as_int(dist)); // rappresenta il float come un intero per l'operazione atomica (funziona solo se il float non è un numero negativo)
    }

    __syncthreads();

    if(tid == 0) {
         atomicMax((int*)maxDist, __float_as_int(localMax)); // rappresenta il float come un intero per l'operazione atomica
    }

}



int main(int argc, char* argv[])
{

    
	// Print GPU information
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("\n\tGPU Info:\n");
	printf("\tMax thread per block: %d\n", prop.maxThreadsPerBlock);
	printf("\tMax thread per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("\tWarp size: %d\n", prop.warpSize);
	printf("\tMax shared memory per block: %zu bytes | %zu KB\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024);

	//START CLOCK***************************************
	cudaEvent_t start, stop;
    float elapsedTime;

    CHECK_CUDA_CALL(cudaEventCreate(&start));
    CHECK_CUDA_CALL(cudaEventCreate(&stop));
	
	CHECK_CUDA_CALL(cudaEventRecord(start));
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
		fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [times file]\n");
		fflush(stderr);
		exit(-1);
	}

	char *filename_time = argv[7];

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
	CHECK_CUDA_CALL(cudaEventRecord(stop));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));
    CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\nTempo di allocazione: %f ms\n", elapsedTime);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//**************************************************
	//START CLOCK***************************************
	CHECK_CUDA_CALL(cudaEventRecord(start));

	//**************************************************
	char *outputMsg = (char *)calloc(10000,sizeof(char));
	char line[100];

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

	cudaEvent_t startKernelOne, stopKernelOne;
    float elapsedTimeKernelOne;
	float kernelOneTotalTime = 0.0f;
	CHECK_CUDA_CALL(cudaEventCreate(&startKernelOne));
	CHECK_CUDA_CALL(cudaEventCreate(&stopKernelOne));


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	dim3 blockDim(64);
	dim3 gridDim1((lines + blockDim.x - 1) / blockDim.x);
	dim3 gridDim2((K * samples + blockDim.x - 1) / blockDim.x);
	dim3 gridDim3((K + blockDim.x - 1) / blockDim.x);

    // attenzione: la shared memory è limitata. ( 40 * 100 + 255 * 100 ) * 4 = 1600 + 102400 = 104000 bytes = 101.5625 KB sulla mia gpu ha 48 KB
    int sharedMemSize_1 = sizeof(int) + (K * samples * sizeof(float)) + (blockDim.x * samples * sizeof(float));

	// Allocazione memoria GPU
	float *d_data, *d_auxCentroids, *d_centroids, *d_maxDist, *d_distCentroids;
	int *d_classMap, *d_changes, *d_pointsPerClass;

	// alloca memoria sulla GPU
	CHECK_CUDA_CALL(cudaMalloc(&d_data, lines * samples * sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc(&d_auxCentroids, K * samples * sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc(&d_centroids, K * samples * sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc(&d_classMap, lines * sizeof(int)));
	CHECK_CUDA_CALL(cudaMalloc(&d_changes, sizeof(int)));
	CHECK_CUDA_CALL(cudaMalloc(&d_pointsPerClass, K * sizeof(int)));
	CHECK_CUDA_CALL(cudaMalloc((void**)&d_maxDist, sizeof(float)));
	CHECK_CUDA_CALL(cudaMalloc((void**)&d_distCentroids, K * sizeof(float)));
	
	// Copia dati sulla GPU
	CHECK_CUDA_CALL(cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA_CALL(cudaMemcpy(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA_CALL(cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice));

	// N.B: Cerca di ridurre il l'uso di cudaMemcpy Host <-> Device nel do-while (molto lento)
	do{
		it++;
		
		//1. Calculate the distance from each point to the centroid
		//Assign each point to the nearest centroid.

		// Aggiorna i dati sulla GPU
		CHECK_CUDA_CALL(cudaMemset(d_changes,0, sizeof(int)));
		CHECK_CUDA_CALL(cudaMemset(d_maxDist, 0, sizeof(float)));
        // reset arrays
		CHECK_CUDA_CALL(cudaMemset(d_pointsPerClass, 0, K*sizeof(int))); //zeroIntArray(pointsPerClass,K);
		CHECK_CUDA_CALL(cudaMemset(d_auxCentroids, 0, K * samples * sizeof(float))); //zeroFloatMatriz(auxCentroids,K,samples);

		// prendere il tempo del primo kernel
		CHECK_CUDA_CALL(cudaEventRecord(startKernelOne));

		KernelClusterAssignment<<<gridDim1, blockDim, sharedMemSize_1>>>(d_data, d_centroids, d_classMap, d_changes, lines, K, samples, d_pointsPerClass, d_auxCentroids);
		CHECK_CUDA_LAST();
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_LAST();

		CHECK_CUDA_CALL(cudaEventRecord(stopKernelOne));
		CHECK_CUDA_CALL(cudaEventSynchronize(stopKernelOne)); 
		CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsedTimeKernelOne, startKernelOne, stopKernelOne));
		kernelOneTotalTime += elapsedTimeKernelOne;

		// Copia il numero di cambiamenti dalla GPU
		CHECK_CUDA_CALL(cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost)); // lento

		// Questo ciclo aggiorna i centroidi calcolando la media delle coordinate dei punti assegnati a ciascun cluster.
		KenelUpdateCentroids<<<gridDim2, blockDim>>>(d_auxCentroids, d_pointsPerClass, K, samples);
		CHECK_CUDA_LAST();
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_LAST();
		// Questo ciclo calcola la massima distanza tra i centroidi
		KernelMaxDistance<<<gridDim3, blockDim>>>(d_centroids, d_auxCentroids, K, samples, d_maxDist);
		CHECK_CUDA_LAST();
		CHECK_CUDA_CALL(cudaDeviceSynchronize());
		CHECK_CUDA_LAST();
		CHECK_CUDA_CALL(cudaMemcpy(d_centroids, d_auxCentroids, K * samples * sizeof(float), cudaMemcpyDeviceToDevice));
		CHECK_CUDA_CALL(cudaMemcpy(&maxDist, d_maxDist, sizeof(float), cudaMemcpyDeviceToHost)); // lento

		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = line;

	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

	CHECK_CUDA_CALL(cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost)); // lento ma lo esegue solo una volta (fuori dal do-while)

	printf("Kernel 1 - Total time: %f s\n", (kernelOneTotalTime / 1000));

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	CHECK_CUDA_CALL(cudaEventRecord(stop));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));
    CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\nComputation time: %f ms\n", elapsedTime);

	error = writeTime(elapsedTime, filename_time);
	if (error != 0) {
			fprintf(stderr, "Error writing time to file");
			exit(1);
	}

	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	CHECK_CUDA_CALL(cudaEventRecord(start));
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

	// free GPU-Memory
	cudaFree(d_data);
	cudaFree(d_centroids);
	cudaFree(d_classMap);
	cudaFree(d_changes);
	cudaFree(d_pointsPerClass);
	cudaFree(d_auxCentroids);
	cudaFree(d_maxDist);
	cudaFree(d_distCentroids);

	//END CLOCK*****************************************
	CHECK_CUDA_CALL(cudaEventRecord(stop));
    CHECK_CUDA_CALL(cudaEventSynchronize(stop));
    CHECK_CUDA_CALL(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\nTempo di deallocazione: %f ms\n", elapsedTime);
	fflush(stdout);
	//***************************************************/
	return 0;
}

//** 
// Come compilare sul cluster sapienza
// srun --partition=students --gpus=1 nvcc -lm -fmad=false -arch=sm_75 KMEANS_cuda.cu -o KMEANS_cuda
// srun --partition=students --gpus=1 KMEANS_cuda test_files/input100D2.inp 30 500 0.1 0.1 output_cuda.txt times_cuda.txt
//  */



