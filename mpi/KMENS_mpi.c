/*
* k-Means clustering algorithm
*
* MPI version
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
#include <mpi.h>
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


int writeTime(float t, const char* filename) {

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
float euclideanDistanceSquared(float *point, float *center, int samples)
{
        float dist=0.0;
        for(int i=0; i<samples; i++)
        {
                dist+= (point[i]-center[i])*(point[i]-center[i]);
        }

        return(dist);
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
        // ho messo 16 perche un float è 4 byte e 16*4 = 64 byte che è la dimensione di una cache line
        // comando per vedere la lunghezza di una cache line: getconf LEVEL1_DCACHE_LINESIZE
        #pragma omp for  schedule(static, 16)
        for (int k=0; k<rows*columns; k++){
                matrix[k] = 0.0;
        }
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
        int i;
        #pragma omp for schedule(static, 16)
        for (i=0; i<size; i++)
                array[i] = 0;
}

void cpyarray(float* A, float* B, int size)
{
        #pragma omp for schedule(static, 16)
        for (int i = 0; i < size; i++)
        {
                A[i] = B[i];
        }
}

int main(int argc, char* argv[])
{
        /* 0. Initialize MPI */
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        if (provided != MPI_THREAD_FUNNELED) {
                fprintf(stderr, "MPI does not provide the required threading level.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int rank, size;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        MPI_Comm_size( MPI_COMM_WORLD, &size );
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

        //START CLOCK***************************************
        double start, end;
        start = MPI_Wtime();
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
        * argv[7]: Number of threads
        * argv[8]: Output file for time
        * */
        if(argc !=  9)
        {
                fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
                fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [number of threads] [time.txt]\n");
                fflush(stderr);
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }

        char *filename = argv[8];

        omp_set_num_threads(atoi(argv[7]));

        // Reading the input data
        // lines = number of points; samples = number of dimensions per point
        int lines = 0, samples= 0;

        int error = readInput(argv[1], &lines, &samples);
        if(error != 0)
        {
                showFileError(error,argv[1]);
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }

        float *data = (float*)calloc(lines*samples,sizeof(float));
        if (data == NULL)
        {
                fprintf(stderr,"Memory allocation error.\n");
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }
        error = readInput2(argv[1], data);
        if(error != 0)
        {
                showFileError(error,argv[1]);
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }

        // Initial centrodis
        srand(0);
        int i;
        for(i=0; i<K; i++)
                centroidPos[i]=rand()%lines;

        // Loading the array of initial centroids with the data from the array data
        // The centroids are points stored in the data array.
        initCentroids(data, centroids, centroidPos, samples, K);

        if(rank == 0) {
                printf("\n\tNumber of threads for one process: %d\n", omp_get_max_threads());
                printf("\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
                printf("\tNumber of clusters: %d\n", K);
                printf("\tMaximum number of iterations: %d\n", maxIterations);
                printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
                printf("\tMaximum centroid precision: %f\n", maxThreshold);
        }

        //END CLOCK*****************************************
        end = MPI_Wtime();;
        double allocTime = end - start;
        MPI_Reduce(&allocTime, &allocTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(rank == 0) {
                printf("\nMemory allocation: %f seconds\n", allocTime);
                fflush(stdout);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        //**************************************************
        //START CLOCK***************************************

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
                MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
        }
/*
ANNOTAZIONI:
        [assign2cluster]
                è possibile suddividere il lavoro, perche il calcolo è indipendente,
                ma bisogna fare la somma totale di changes.
                ho rimosso la funzione calculateDistance perche potrebbe produrre overhead di chiamata.
                ho provato ad usare SIMD nel terzo for
                        // #pragma omp simd
                        // for(int d=0; d<samples; d++)
                        // {
                        //      vector_dist[d] = (point[d]-center[d])*(point[d]-center[d]);
                        // }
                        // for(int d=0; d<samples; d++)
                        // {
                        //      dist += vector_dist[d];
                        // }
                        ma aumenta il tempo di esecuzione
        [calculateCentroids]
                Quando dividi il carico di lavoro tra i processi,
                ciascun processo può calcolare parzialmente i contributi per il ricalcolo dei centroidi
                (cioè, la somma delle coordinate dei punti e il conteggio dei punti per ogni cluster)
                riferiti solo al suo sottoinsieme di dati.
                Tuttavia, i processi non sono completamente indipendenti in quanto, per ottenere i nuovi
                centroidi corretti, è necessario riunire (o ridurre) questi contributi parziali provenienti
                da tutti i processi.
        [maxDist]

        [post do-while]
                riassemblo classmap tutti nel processo 0 cosi puo scriverli sul file


*/


/*
*
* START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
*
*/
        // Divido il numero di punti in modo equo tra i processi
        int baseNumPoints = lines / size;
        int extraPoints = lines % size;
        int startIndex = rank * baseNumPoints + (rank < extraPoints ? rank : extraPoints);
        int endIndex = startIndex + baseNumPoints + (rank < extraPoints ? 1 : 0);
        int localLines = endIndex - startIndex;

        int* localClassMap = (int*)calloc(localLines, sizeof(int));
        int *localpointsPerClass = (int*)calloc(K,sizeof(int));
        float *localauxCentroids = (float*)calloc(K*samples, sizeof(float));


        // COMPUTAZIONE
        start = MPI_Wtime();
        do{
                it++;

                //1. Calculate the distance from each point to the centroid
                //Assign each point to the nearest centroid.
                changes = 0;
                #pragma omp parallel for private(i,j,class,dist,minDist) reduction(+:changes) schedule(dynamic)
                for(i=0; i<localLines; i++)
                {
                        class=1;
                        minDist=FLT_MAX;
                        for(j=0; j<K; j++)
                        {
                                dist=0.0;
                                float *point = &data[(startIndex + i)*samples];
                                float *center = &centroids[j*samples];

                                for(int d=0; d<samples; d++)
                                {
                                        dist+= (point[d]-center[d])*(point[d]-center[d]);
                                }
                                dist = sqrt(dist);

                                if(dist < minDist)
                                {
                                        minDist=dist;
                                        class=j+1;
                                }
                        }
                        if (localClassMap[i]!=class)
                        {
                                changes++;
                        }
                        localClassMap[i]=class;
                }
                MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

                // 2. Recalculates the centroids: calculates the mean within each cluster
                #pragma omp parallel
                {
                        zeroIntArray(pointsPerClass,K);
                        zeroIntArray(localpointsPerClass,K);
                        zeroFloatMatriz(auxCentroids,K,samples);
                        zeroFloatMatriz(localauxCentroids,K,samples);

                        #pragma omp barrier

                        #pragma omp for reduction(+: localpointsPerClass[0:K]) reduction(+: localauxCentroids[0:K*samples])
                        for (int i = 0; i < localLines; i++) {
                                int cls = localClassMap[i];
                                localpointsPerClass[cls - 1]++;
                                for (int j = 0; j < samples; j++) {
                                        localauxCentroids[(cls - 1) * samples + j] += data[(i + startIndex) * samples + j];
                                }
                        }
                }

                // aggregazione globale (la riduzione) per combinare i risultati parziali.
                MPI_Allreduce(localpointsPerClass, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(localauxCentroids, auxCentroids, K*samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                #pragma omp parallel shared(maxDist)
                {
                        // per questo for si necessitano gli array auxCentroids e pointsPerClass completi!
                        #pragma omp for private(i,j) schedule(static)
                        for(int t=0; t<K*samples; t++)
                        {
                                i = t / samples;
                                j = t - i*samples;
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
                        cpyarray(centroids, auxCentroids, K*samples);
                }

                sprintf(line,"\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
                //outputMsg = strcat(outputMsg,line);
                outputMsg = line;

                MPI_Barrier(MPI_COMM_WORLD);
        } while((changes>minChanges) && (it<maxIterations) && (maxDist>maxThreshold));

        int recvcounts[size];  // Numero di elementi per ogni processo
        int displs[size];      // Offset di inizio per ogni processo

        // Calcola recvcounts e displs da usare in MPI_Allgatherv
        displs[0] = 0;
        for (int i = 0; i < size; i++) {
                recvcounts[i] = (lines / size) + (i < (lines % size) ? 1 : 0);
                if (i > 0) displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
        // Eseguire la raccolta dei dati locali in classMap globale di rank 0 da salvare sul file
        MPI_Gatherv(localClassMap, localLines, MPI_INT, classMap, recvcounts, displs, MPI_INT, 0 ,MPI_COMM_WORLD);
/*
*
* STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
*
*/
        //END CLOCK*****************************************
        end = MPI_Wtime();
        double computeTime = end - start;
        MPI_Reduce(&computeTime, &computeTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0){
                // Output and termination conditions
                printf("%s",outputMsg);
                printf("\nComputation: %f seconds", computeTime);
                printf("\nRank %d", rank);
                fflush(stdout);

                // Write time to file
                error = writeTime(computeTime, filename);
                if (error != 0) {
                        fprintf(stderr, "Error writing time to file: %s\n", filename);
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
        }

        //**************************************************
        //START CLOCK***************************************
        start = MPI_Wtime();
        //**************************************************


        // Writing the classification of each point to the output file.
        if (rank == 0) {

                if (changes <= minChanges) {
                        printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
                }
                else if (it >= maxIterations) {
                        printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
                }
                else {
                        printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
                }

                error = writeResult(classMap, lines, argv[6]);
                if(error != 0)
                {
                        showFileError(error, argv[6]);
                        MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
                }
        }


        free(localauxCentroids);
        free(localClassMap);
        free(localpointsPerClass);

        //Free memory
        free(data);
        free(classMap);
        free(centroidPos);
        free(centroids);
        free(distCentroids);
        free(pointsPerClass);
        free(auxCentroids);

        //END CLOCK*****************************************
        end = MPI_Wtime();
        double deallocTime = end - start;
        MPI_Reduce(&deallocTime, &deallocTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
                printf("\n\nMemory deallocation: %f seconds\n", deallocTime);
                fflush(stdout);
        }
        //***************************************************/
        MPI_Finalize();
        return 0;
}