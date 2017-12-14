#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N  2048


double maxeps = 0.1e-7;
int itmax = 100;
double w = 0.5;
double eps;
int ranksize, myrank;
int min_str = 0, max_str = N-1;


double A [N][N];

void relax();
void init();
void verify(); 
void send_matrixpart();
void get_matrix();
int get_minstr(int, int);
int get_maxstr(int, int);

int main(int argc, char **argv){
	double t1, t2;
	MPI_Init(&argc, &argv);
	init();
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Barrier(MPI_COMM_WORLD);
	if (myrank == 0){
		t1 = MPI_Wtime();
	}

	min_str = get_minstr(ranksize, myrank);
	max_str = get_maxstr(ranksize, myrank);

	for(int it = 1; it <= itmax; ++it){
		eps = 0.;
		relax();
		MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (eps < maxeps) 
			break;
	}

	if (myrank != 0){
		send_matrixpart();
	} else {
		get_matrix();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (myrank == 0){
		t2 = MPI_Wtime();
		printf("time  = %f\n", t2 - t1);
		verify();
	}
	MPI_Finalize();
	return 0;
}

void init(){ 
	for(int i = 0; i <= N-1; i++){
		for(int j = 0; j <= N-1; j++){
			if(i == 0 || i == N-1 || j == 0 || j == N-1){ 
				A[i][j]= 0.;
			} else { 
				A[i][j]= ( 1. + i + j);
			}
		}
	}
} 

void relax(){
	MPI_Status status;
	MPI_Request request;
	int nextrank = myrank == ranksize - 1 ? MPI_PROC_NULL : myrank + 1;
	int prevrank = myrank == 0 ? MPI_PROC_NULL : myrank - 1;
	for(int i = min_str + 1; i <= max_str - 1; i++){
		for(int j = 1 + (i % 2); j <= N-2; j += 2){
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			A[i][j] = A[i][j] + b;
			eps = Max(fabs(b),eps);			
		}
	}

	MPI_Isend(A[max_str-1], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);
	
	MPI_Isend(A[min_str + 1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[min_str], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);

	for(int i = min_str + 1; i <= max_str - 1; ++i){
		for(int j = 1 + ((i + 1) % 2); j <= N-2; j += 2){
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			A[i][j] = A[i][j] + b;
		}
	}

	MPI_Isend(A[max_str - 1], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[max_str], N, MPI_DOUBLE, nextrank, 0, MPI_COMM_WORLD, &status);

	MPI_Isend(A[min_str + 1], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &request);
	MPI_Recv(A[min_str], N, MPI_DOUBLE, prevrank, 0, MPI_COMM_WORLD, &status);
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

void send_matrixpart(){
	MPI_Send(A[min_str + 1], (max_str - min_str - 1)*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

int get_maxstr(int num, int rank){
	if (rank == num - 1){
		return N-1;
	} else {
		return (N / num)*(rank + 1);
	}
}

int get_minstr(int num, int rank){
	if (rank == 0){
		return 0;
	} else {
		return N / num * rank - 1;
	}
}
void get_matrix(){
	MPI_Status status;
	for (int i = 1; i < ranksize; ++i){
		int tmp_min = get_minstr(ranksize, i);
		int tmp_max = get_maxstr(ranksize, i);
		MPI_Recv(A[tmp_min + 1], (tmp_max - tmp_min- 1)*N, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
	}
}