#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <mpi.h>
 
#define eps 0.1e-7
double w = 0.5;

MPI_Request req[4];
MPI_Status status;

int myrank, ranksize;
int startcol, lastcol, nextrank, prevrank;

int N;
double **A;
double *buf[4];

void range(int n1, int n2, int nprocs, int myrank, int *startrow, int *lastrow);
void exchange(int phase);
void verify();

void init(double **A, int N) {
 
    for (int i = 0; i <= N - 1; i++){
        for (int j = 0; j <= N - 1; j++){
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1)
                A[i][j]= 0.;
            else
                A[i][j]= (1. + i + j) ;
        }
    }
}

void matrixAlloc(double ***A, int N) {
    int i;
    double** tmpA = (double**)malloc(N*sizeof(double*));
    for (i = 0; i < N; i++)
        tmpA[i] = (double*)malloc(N*sizeof(double));
    *A = tmpA;
}
 
void bufAlloc() {
	for (int i = 0; i < 4; ++i){
		buf[i] = (double *)malloc(N * sizeof (double));
	}
}
 
int main (int argc, char **argv) {
    
    double newA, r, w, stopdiff, maxdiff, diff;
    int i, j, phase, iteration;
    double start, end;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
 
    N = atoi(argv[1]);
 
    stopdiff = eps;
 
    matrixAlloc(&A, N);
    init(A, N);
    bufAlloc();
 
    range(0, N, ranksize, myrank, &startcol, &lastcol);
 
    nextrank = myrank + 1;
    prevrank = myrank - 1;
 
    if (nextrank == ranksize)
        nextrank = MPI_PROC_NULL;
 
    if (prevrank == -1)
        prevrank = MPI_PROC_NULL;
 
    if (myrank == 0) {
        start = MPI_Wtime();
    }
 
    do {
        maxdiff = 0.0;
        for (phase = 0; phase < 2; phase++){
            exchange(phase);
            for (i = 1 ; i < N - 1 ; i++) {
                for (j = startcol + (!(i & 1) ^ phase); j <= lastcol ; j += 2) {
                    newA = (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]) / 4.0;
                    diff = fabs(newA - A[i][j]);
                    if (diff > maxdiff)
                        maxdiff = diff;
                    A[i][j] = A[i][j] + w * (newA - A[i][j]);
                }
            }
        }
        MPI_Allreduce(&maxdiff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } while (maxdiff < stopdiff);
 
    if(myrank == 0) {
        end = MPI_Wtime();
        printf("%f\n", end - start);
        verify();
    }
 
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
 
    return 0;
}
 

 

 
void range(int n1, int n2, int nprocs, int myrank, int *startrow, int *lastrow) {
 
    float iwork1 = (n2 - n1 + 1) / nprocs;
    float iwork2 = (n2 - n1 + 1) % nprocs;
 
    int start, end;
 
    start = (int) (myrank * iwork1 + n1 + (myrank < iwork2? myrank : iwork2));
 
    end = (int) (start + iwork1 - 1);
 
    if (iwork2 > myrank) {
        end = end + 1;
    }
 
    *startrow = start;
    *lastrow = end;
}
 
void exchange(int phase) {
 
    int is1 = ((startcol + phase) % 2) + 1;
    int is2 = ((lastcol + phase) % 2) + 1;
    int ir1 = fabs(3 - is1);
    int ir2 = fabs(3 - is2);
    int icnt = 0;
    int icnt1 = 0, icnt2 = 0;
    int m = N - 1;
    int i;
 
    if (myrank != 0) {
        icnt1 = 0;
        for (i = is1; i <= m; i += 2) {
            buf[0][icnt1] = A[i][startcol];
            icnt1 = icnt1 + 1;
        }
    }
 
    if (myrank != (ranksize - 1)) {
        icnt2 = 0;
        for (i = is2; i <= m; i += 2) {
            buf[1][icnt2] = A[i][lastcol];
            icnt2 = icnt2 + 1;
        }
    }
 
    MPI_Isend(buf[0], icnt1, MPI_DOUBLE, prevrank, 1, MPI_COMM_WORLD, &req[0]);
    MPI_Isend(buf[1], icnt2, MPI_DOUBLE, nextrank, 1, MPI_COMM_WORLD, &req[1]);
 
    MPI_Wait(&req[0], &status);
    MPI_Wait(&req[1], &status);
 
    MPI_Irecv(buf[2], N, MPI_DOUBLE, prevrank, 1, MPI_COMM_WORLD, &req[2]);
    MPI_Irecv(buf[3], N, MPI_DOUBLE, nextrank, 1, MPI_COMM_WORLD, &req[3]);
 
   
    MPI_Wait(&req[2], &status);
    MPI_Wait(&req[3], &status);
 
    if (myrank != 0) {
        icnt = 0;
        for (i = ir1; i <= m; i += 2) {
            A[i][startcol - 1] = buf[2][icnt];
            icnt = icnt + 1;
        }
    }
 
    if (myrank != (ranksize - 1)) {
        icnt = 0;
        for (i = ir2; i <= m; i += 2) {
            A[i][lastcol + 1] = buf[3][icnt];
            icnt = icnt + 1;
        }
    }
}
 
void verify() {
    double s;
    s = 0.;
    for(int i = 0; i <= N - 1; i++){
        for(int j = 0; j <= N - 1; j++){
            s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }
    printf("S = %f\n",s);
}
