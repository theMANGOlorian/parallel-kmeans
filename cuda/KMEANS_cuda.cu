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

__global__ void KernelClusterAssignment(float* data, float* centroids, int* classMap, int* changes, int lines, int K, int samples) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x; // Indice del punto corrente
    int tid = threadIdx.x;
	__shared__ int sharedData[256];

    float dist;
    if (pointIdx < lines) {
        float minDist = FLT_MAX; // Distanza minima iniziale
        int closestClass = 0;   // Classe più vicina

        for (int j = 0; j < K; j++) { // Ciclo sui centroidi
            dist = CudaEuclideanDistance(&data[pointIdx * samples], &centroids[j * samples], samples); // Calcolo distanza

            if (dist < minDist) { // Aggiornamento se il centroide corrente è più vicino
                minDist = dist;
                closestClass = j + 1;
            }
        }

        // Controllo cambiamenti nella classe
        if (classMap[pointIdx] != closestClass) {
            sharedData[tid] = 1; // Segna che c'è stato un cambiamento
        } else {
            sharedData[tid] = 0; // No cambiamento
        }
        classMap[pointIdx] = closestClass; // Assegna la classe più vicina
    } else {
        sharedData[tid] = 0; // Assicurati che i thread inattivi non influiscano sulla somma
    }

    __syncthreads();  // Sincronizzazione tra i thread per evitare conflitti durante la riduzione

	// strided reduction
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // Ogni thread somma il valore a un altro thread a distanza "stride"
        if (tid % (2 * stride) == 0 && (tid + stride) < blockDim.x) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();  // Sincronizza i thread
    }

    // Solo il thread 0 di ogni blocco aggiorna la variabile globale
    if (tid == 0) {
        atomicAdd(changes, sharedData[0]);  // Somma il contatore del blocco alla variabile globale
    }
}

// da ottimizare
__global__ void KernelComputeClusterSum(float* data, int* classMap, float* auxCentroids, int* pointsPerClass, int lines, int samples, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Assicurati che i thread non escano dai limiti
    if (i < lines) {
        int cluster = classMap[i];  // Ottieni la classe del punto i
        atomicAdd(&pointsPerClass[cluster - 1], 1);  // Incrementa il numero di punti per la classe (atomic per evitare condizioni di race)

        for (int j = 0; j < samples; j++) {
            // Somma i valori dei dati al centroide corrispondente
            atomicAdd(&auxCentroids[(cluster - 1) * samples + j], data[i * samples + j]);
        }
    }
}

__global__ void KenelUpdateCentroids(float* auxCentroids, int* pointsPerClass, int K, int samples){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < K*samples) {
		int i = idx / samples;
		int j = idx - i * samples;
		auxCentroids[i*samples+j] /= pointsPerClass[i];
	}
}

__global__ void KernelMaxDistance(float* centroids, float* auxCentroids, float* distCentroids, int K, int samples, float *maxDist){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Memoria condivisa per il massimo all'interno di un blocco
    __shared__ float sharedMaxDist[256];
    
    int tid = threadIdx.x;

    // Inizializza sharedMaxDist con il valore calcolato dalla distanza
    if (i < K) {
        distCentroids[i] = CudaEuclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
    } else {
        distCentroids[i] = FLT_MIN;  // Imposta il valore di default per i thread fuori range
    }

    // Ogni thread salva la propria distanza nella memoria condivisa
    sharedMaxDist[tid] = distCentroids[i];
	// imposta il valore massimo a FLT_MIN
	if (i == 0) {
		*maxDist = FLT_MIN;
	}
    __syncthreads();  // Sincronizza tutti i thread

    // Fase di riduzione: trovare il massimo all'interno del blocco [ s>>=1 è una divisione (intera) per 2 ]
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedMaxDist[tid] = fmaxf(sharedMaxDist[tid], sharedMaxDist[tid + s]);
        }
        __syncthreads();  // Sincronizza i thread dopo ogni passo di riduzione
    }

    // Il thread 0 scrive il massimo del blocco nella variabile globale maxDist
    if (tid == 0) {
        atomicMax((int*)maxDist, __float_as_int(sharedMaxDist[0]));
    }
}



int main(int argc, char* argv[])
{

	//START CLOCK***************************************
	cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
	cudaEventRecord(start);
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
	if(argc !=  7)
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
	maxThreshold = maxThreshold * maxThreshold;

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
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nTempo di allocazione: %f ms\n", elapsedTime);
	fflush(stdout);

	CHECK_CUDA_CALL( cudaSetDevice(0) );
	CHECK_CUDA_CALL( cudaDeviceSynchronize() );
	//**************************************************
	//START CLOCK***************************************
	cudaEventRecord(start);

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

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	// Numero di thread per blocco e calcolo dei blocchi
	int threadsPerBlock = 256;
	int blocksPerGrid = (lines + threadsPerBlock - 1) / threadsPerBlock;
	int blocksPerGrid2 = (K * samples + threadsPerBlock - 1) / threadsPerBlock;
	int blocksPerGrid3 = (K + threadsPerBlock - 1) / threadsPerBlock;

	printf("\nThreads per block: %d",threadsPerBlock);
	printf("\nBlocks per grid: %d",blocksPerGrid);

	// Allocazione memoria GPU
	float *d_data, *d_auxCentroids, *d_centroids, *d_maxDist, *d_distCentroids;
	int *d_classMap, *d_changes, *d_pointsPerClass;

	// alloca memoria sulla GPU
	cudaMalloc(&d_data, lines * samples * sizeof(float));
	cudaMalloc(&d_auxCentroids, K * samples * sizeof(float));
	cudaMalloc(&d_centroids, K * samples * sizeof(float));
	cudaMalloc(&d_classMap, lines * sizeof(int));
	cudaMalloc(&d_changes, sizeof(int));
	cudaMalloc(&d_pointsPerClass, K * sizeof(int));
	cudaMalloc((void**)&d_maxDist, sizeof(float));
	cudaMalloc((void**)&d_distCentroids, K * sizeof(float));
	
	// Copia dati sulla GPU
	cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);

	// N.B: Cerca di ridurre il l'uso di cudaMemcpy Host <-> Device nel do-while (molto lento)
	do{
		it++;
		
		//1. Calculate the distance from each point to the centroid
		//Assign each point to the nearest centroid.

		// Aggiorna i dati sulla GPU
		cudaMemset(d_changes,0, sizeof(int));
		KernelClusterAssignment<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_centroids, d_classMap, d_changes, lines, K, samples);
		cudaDeviceSynchronize();
		
		// Copia il numero di cambiamenti dalla GPU
		cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost); // lento

		// 2. Recalculates the centroids: calculates the mean within each cluster
		cudaMemset(d_pointsPerClass, 0, K*sizeof(int)); //zeroIntArray(pointsPerClass,K);
		cudaMemset(d_auxCentroids, 0, K * samples * sizeof(float)); //zeroFloatMatriz(auxCentroids,K,samples);
		
		// Questo ciclo serve a calcolare la somma dei punti appartenenti a ciascun cluster.
		KernelComputeClusterSum<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_classMap, d_auxCentroids, d_pointsPerClass, lines, samples, K);
		cudaDeviceSynchronize();
		
		// Questo ciclo aggiorna i centroidi calcolando la media delle coordinate dei punti assegnati a ciascun cluster.
		KenelUpdateCentroids<<<blocksPerGrid2, threadsPerBlock>>>(d_auxCentroids, d_pointsPerClass, K, samples);
		cudaDeviceSynchronize();
		// Questo ciclo calcola la massima distanza tra i centroidi
		KernelMaxDistance<<<blocksPerGrid3, threadsPerBlock>>>(d_centroids, d_auxCentroids, d_distCentroids, K, samples, d_maxDist);
		cudaDeviceSynchronize();
		cudaMemcpy(d_centroids, d_auxCentroids, K * samples * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&maxDist, d_maxDist, sizeof(float), cudaMemcpyDeviceToHost); // lento

		sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg,line);

	} while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

	cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost);
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */
	// Output and termination conditions
	printf("%s",outputMsg);	

	CHECK_CUDA_CALL( cudaDeviceSynchronize() );

	//END CLOCK*****************************************
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nComputation time: %f ms\n", elapsedTime);
	fflush(stdout);
	//**************************************************
	//START CLOCK***************************************
	cudaEventRecord(start);
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
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nTempo di deallocazione: %f ms\n", elapsedTime);
	fflush(stdout);
	//***************************************************/
	return 0;
}