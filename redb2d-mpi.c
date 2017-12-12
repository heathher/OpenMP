#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   1000
double   maxeps = 0.1e-7;
int itmax = 1000;
MPI_Status status[4];
int ll, shift;
double w = 0.5;
double eps;
MPI_Request req[4];

double A [N][N];

void relax();
void init();
void verify(); 


int main(int argc, char **argv)
{
	int myrank, ranksize;
	int it;
	double t1, t2;
	int startrow, lastrow, nrow;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
	MPI_Barrier(MPI_COMM_WORLD);
	startrow = (myrank * N)/ ranksize;
	lastrow = (((myrank + 1) * N) / ranksize) - 1;
	nrow = lastrow - startrow + 1;

	//printf("0");
	//printf("%i %i\n", nrow, startrow);
	if (myrank == 0)
		t1 = MPI_Wtime();

	//printf("*\n");
	//printf("4\n");
	init();
	//printf("3\n");

	//printf("*\n");
	for(it = 1; it <= itmax; it++){
		eps = 0.;

		//printf("*\n");
		relax(nrow, myrank, ranksize);
		//printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) 
			break;
	}

	verify();
	MPI_Barrier(MPI_COMM_WORLD);
	if (myrank == 0){
		t2 = MPI_Wtime();
		printf("%d: Time of task = %lf\n", myrank, t2 - t1);
	}
	MPI_Finalize();
	return 0;
}


void init()
{ 
	for(int i = 0; i <= N-1; i++){
		for(int j = 0; j <= N-1; j++){
			if(i == 0 || i == N-1 || j == 0 || j == N-1){ 
				A[i][j]= 0.;
			} else {
				A[i][j]= ( 1. + i + j) ;
			}
		}
	}
} 


void relax(int nrow, int myrank, int ranksize)
{
	//printf("1\n");

	//printf("ss%d \n",nrow);
	for(int i = 1; i <= nrow-2; i++){
		if (((i==1)&&(myrank==0)) || ((i==nrow) &&(myrank == ranksize -1)))	
				continue;
		for(int j = 1 + i % 2; j <= N-2; j+=2){
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			eps =  Max(fabs(b),eps);
			A[i][j] = A[i][j] + b;
		}
	}
	//printf("2\n");
	//printf("1\n");
	if (myrank != 0)
		MPI_Irecv(&A[0][0],N,MPI_DOUBLE, myrank-1,1000,MPI_COMM_WORLD, &req[0]);

	//printf("2\n");
	if (myrank != ranksize - 1)
		MPI_Isend(&A[nrow][0],N,MPI_DOUBLE, myrank+1,1000,MPI_COMM_WORLD, &req[2]);


	//printf("3\n");

	if (myrank != ranksize - 1)
		MPI_Irecv(&A[nrow+1][0],N,MPI_DOUBLE, myrank+1,1001,MPI_COMM_WORLD, &req[3]);

	//printf("4\n");

	if (myrank != 0)
		MPI_Isend(&A[1][0],N,MPI_DOUBLE, myrank-1,1001,MPI_COMM_WORLD, &req[1]);

	ll = 4;
	shift = 0;
	if (myrank == 0){ 
		ll = 2; 
		shift = 2; 
	}
	if (myrank == ranksize - 1){ 
		ll = 2; 
	}
	//MPI_Waitall(ll, &req[shift], &status[0]);


	for(int i = 1; i <= nrow-2; i++){
		if (((i==1) && (myrank == 0)) || ((i == nrow) && (myrank == ranksize -1))) 
			continue;
		for(int j = 1 + (i + 1) % 2; j<=N-2; j+=2){
			double b;
			b = w*((A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4. - A[i][j]);
			A[i][j] = A[i][j] + b;
		}
	}

}


void verify()
{ 
	double s;

	s=0.;
	for(int i=0; i<=N-1; i++){
		for(int j=0; j<=N-1; j++){
			s=s+A[i][j]*(i+1)*(j+1)/(N*N);
		}
	}
	//printf("  S = %f\n",s);
}