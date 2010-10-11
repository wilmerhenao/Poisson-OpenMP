/* ---------------------------------------------------------------------

   solve 3d poisson equation   lapl u = f  u = g on bndry
   in square domain  [xlo, xhi] by [ylo, yhi] by [zlo, zhi] (set to [-1, 1])
   with dirichlet boundary conditions using simple jacobi iteration

   To run: type
     xserial   nx ny nz
   where nx,ny,nz are the number of grid points in each dimension
   
   For example, if nx = 5 on [-1,1] including the boundary conditions
   then there are 4 cells, so  dx = .5, and there are 3 unknowns
   since there are 2 boundary conditions that are known.

-------------------------------------------------------------------- */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

#include <time.h>
#include "timing.h"

/* ----- local prototypes -------------------------------------------------- */

int main ( int argc, char *argv[] );
int no_timing(int argc, int argv[3]);
void get_prob_size(int *nx, int *ny, int *nz, int argc, int* argv);
void driver ( int nx, int ny, int nz, long int it_max, double tol,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval );
void jacobi ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval );
void gauss_seidel ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval, char rb[] );
void SOR ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval, char rb[] );
void init_prob ( int nx, int ny, int nz, double f[], double u[],
                 double xlo, double ylo, double zlo,
                 double xhi, double yhi, double zhi );
void calc_error ( int nx, int ny, int nz, double u[], double f[],
                  double xlo, double ylo, double zlo,
                  double xhi, double yhi, double zhi  );
double u_exact   ( double x, double y, double z );
double uxx_exact ( double x, double y, double z );
double uyy_exact ( double x, double y, double z );
double uzz_exact ( double x, double y, double z );
void rb_vector(char rb[], int nx, int ny, int nz);
/* ----- local macros  -------------------------------------------------- */

#define INDEX3D(i,j,k)    ((i)+(j)*nx + (k)*(nx)*(ny))  //NOTE: column major (fortran-style) ordering here
#define MAX(A,B)          ((A) > (B)) ? (A) : (B)
#define ABS(A)            ((A) >= 0) ? (A) : -(A)
#define F(i,j,k)          f[INDEX3D(i,j,k)]
#define U(i,j,k)          u[INDEX3D(i,j,k)]
#define RB(i,j,k)      	  rb[INDEX3D(i,j,k)]
#define U_OLD(i,j,k)      u_old[INDEX3D(i,j,k)]
const int deftrials = 5;

/* ------------------------------------------------------------------------- */

int main( int argc, char *argv[]){
	// This is wrong but I will just ignore and overwrite the arguments
    	int traxh = 0;
	int i, trials;
	int * test_sizes;
	int myargs[3] = {10, 10, 10}; 
	struct timespec start, finish;
	double secs = -1.0;
	FILE *fp;
	if (argc == 1){
		// Assign five trials if nothing else is provided
		trials = deftrials;
		test_sizes = (int *) malloc(trials * sizeof(int));
	    	for(i = 0; i < trials; ++i)
			test_sizes[i] = (i+1) *20;
	} else {
		// Or assign the number of trials provided as an argument
		trials = argc-1;
		test_sizes = (int *) malloc((argc-1) * sizeof(int));
		for (i = 1; i < argc; ++i){
			test_sizes[i-1] = atoi(argv[i]);
		}
	}
	if (fp = fopen("results.dat", "w"));
    	for (i = 0; i < trials; ++i){
		myargs[0] = myargs[1] = myargs[2] = test_sizes[i]; //Create a cube with the same lengths of dx on all sides (just bec. it's easy)
		get_time(&start); // start counting time
       		traxh = no_timing(argc, myargs);
		get_time(&finish); // stop the clock
		secs = timespec_diff(start, finish);
		fprintf (fp, "%u\t%lg\n", myargs[0], secs); // print to the file
    	}
	fclose(fp);
	free(test_sizes);
	return(traxh);
}

int no_timing ( int argc, int argv[3] )
{
  
    int nx = -1;   /* number of grid points in x           */
    int ny = -1;   /*                     and  y direction */
    int nz = -1;   /*                     and  z direction */

    double tol = 1.e-7;     /* convergence criteria */
    long int it_max = 100000;      /* max number of iterations */
    int io_interval =  100; /* output status this often */


    double xlo = -1., ylo = -1., zlo = -1; /* lower corner of domain */
    double xhi =  1., yhi =  1., zhi = 1.; /* upper corner of domain */
  
    
    /* get number of grid points for this experiment */
    get_prob_size(&nx, &ny, &nz, argc, argv);

    //wval1 = omp_get_wtime();
    
    driver ( nx, ny, nz, it_max, tol, xlo, ylo, zlo, xhi, yhi, zhi, io_interval );

    //wval2 = omp_get_wtime();
    //printf("omp walltime  = %15.7e\n",wval2-wval1);
    return 0;
}

/* ------------------------------------------------------------------------- */

void driver ( int nx, int ny, int nz, long int it_max, double tol,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval){
    double *f, *u;
    char * rb;
    int i,j,k;
    double secs = -1.0;
    struct timespec start, finish;

    /* Allocate and initialize  */
    f = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );  
    u = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );
    rb = ( char * ) malloc (nx * ny * nz * sizeof ( char ) );

    get_time( &start );
    for ( k = 0; k < nz; k++ ) 
      for ( j = 0; j < ny; j++ ) 
        for ( i = 0; i < nx; i++ ) 
          U(i,j,k) = 0.0;  /* note use of array indexing macro */

    /* set rhs, and exact bcs in u */
    init_prob ( nx, ny, nz, f, u , xlo, ylo, zlo, xhi, yhi, zhi);
    // rb has the red and black positions
    rb_vector(rb, nx, ny, nz);
    /* Solve the Poisson equation  */
    //jacobi ( nx, ny, nz, u, f, tol, it_max, xlo, ylo, zlo, xhi, yhi, zhi, io_interval );
    //gauss_seidel ( nx, ny, nz, u, f, tol, it_max, xlo, ylo, zlo, xhi, yhi, zhi, io_interval, rb );

    SOR( nx, ny, nz, u, f, tol, it_max, xlo, ylo, zlo, xhi, yhi, zhi, io_interval , rb);
    /* Determine the error  */
    calc_error ( nx, ny, nz, u, f, xlo, ylo, zlo, xhi, yhi, zhi );

    /* get time for initialization and iteration.  */
    get_time(&finish);
    secs = timespec_diff(start,finish);
    printf(" Total time: %15.7e seconds\n",secs);

    free ( u );
    free ( f );
    free (rb);
    return;
}

void rb_vector(char rb[], int nx, int ny, int nz){
    int i;
    
    //first the case when the base is odd | necesito que los dos sean pares o que los dos sean impares
    if(0 == (nx+ny) % 1){
	for(i = 0; i < (nx*ny*nz); ++i){
	    if(i % 1){
		rb[i] = 'r';
	    } else {
	   	rb[i] = 'b';
	    }
	}
    } else {
        int activate = 0;
	for(i = 0; i < (nx*ny*nz); ++i){
	    if((i + activate) % 1){
		rb[i] = 'r';
	    } else {
	   	rb[i] = 'b';
	    }
	    if(0 == (i+1) % (nx * ny) )
		++activate;
	}
	
    //when it's not.  I require a special treatment
	
    }
}

/************************************************************************************/
void SOR ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval, char rb [])
{

    double ax, ay, az, d;
    double dx, dy, dz, rem;
    double update_norm, unew;
    int i, it, j, k, it_used = it_max;
    int istart;
    double *u_old, diff;
    double omega;

    /* Initialize the coefficients.  */

    dx =  (xhi - xlo) / ( double ) ( nx - 1 );
    dy =  (yhi - ylo) / ( double ) ( ny - 1 );
    dz =  (zhi - zlo) / ( double ) ( nz - 1 );

    ax =   1.0 / (dx * dx);
    ay =   1.0 / (dy * dy);
    az =   1.0 / (dz * dz);
    d  = - 2.0 / (dx * dx)  - 2.0 / (dy * dy) -2.0 / (dz * dz);
    omega = 1.0;

    //set the number of threads
    omp_set_num_threads(8);

    for ( it = 1; it <= it_max; it++ ) {
        update_norm = 0.0;

     /* Compute stencil, and update.  bcs already in u. only update interior of domain */
    #pragma omp parallel for default(none)\
    	shared(nx, ny, nz, omega, d, rb, u, f, ax, ay, az) private(i, j, k, rem, diff, istart) reduction(+:update_norm)
      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {

	    istart = 2;
	    if('r' == RB(1,j,k)) // execute the reds first
		istart = 1;
            for ( i = istart; i < nx-1; i+=2 ) {
		rem = U(i,j,k);
                U(i,j,k) = (omega * (F(i,j,k) -
                       	( ax * ( U(i-1,j,k) + U(i+1,j,k) ) +
                       	  ay * ( U(i,j-1,k) + U(i,j+1,k) ) +
                       	  az * ( U(i,j,k-1) + U(i,j,k+1) ) ) ) - (omega - 1.0) * ax * U(i,j,k) ) / d;

                diff = ABS(U(i,j,k)-rem);
                	/*if (diff > update_norm){ using max norm 
                	    update_norm = diff;
                	  }*/
		update_norm += diff*diff;  /* using 2 norm */
            } /* end for i */
        } /* end for j */
      } /* end for k */

      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {
	    istart = 2;
	    if('b' == RB(1,j,k)) // Now execute the blacks
		istart = 1;
            for ( i = istart; i < nx-1; i+=2 ) {
		diff = 0.0;
		rem = U(i,j,k);
               	U(i,j,k) = (omega * (F(i,j,k) -
                       	( ax * ( U(i-1,j,k) + U(i+1,j,k) ) +
                       	  ay * ( U(i,j-1,k) + U(i,j+1,k) ) +
                       	  az * ( U(i,j,k-1) + U(i,j,k+1) ) ) ) - (omega - 1.0) * ax * U(i,j,k) ) / d;

               	diff = ABS(U(i,j,k)-rem);
               
		update_norm += diff*diff;  /* using 2 norm */
                	/*if (diff > update_norm){  using max norm 
                	    update_norm = diff;
                	  }*/
            } /* end for i */
        } /* end for j */
      } /* end for k */

        if (0 == it% io_interval) 
            printf ( " iteration  %5d   norm update %14.4e\n", it, update_norm );

        if ( sqrt(update_norm) <= tol ) {
          it_used = it;
          break;
        }

    } /* end for it iterations */
    printf ( " Final iteration  %5d   norm update %14.6e\n", it_used, update_norm );

    return;
}

/******************************************************************************/

void init_prob ( int nx, int ny, int nz, double  f[], double u[],
                 double xlo,  double ylo, double zlo,
                 double xhi,  double yhi, double zhi )
{
  int    i, j, k;
  double x, y, z, dx, dy, dz;

    dx = (xhi - xlo) / ( double ) ( nx - 1 );
    dy = (yhi - ylo) / ( double ) ( ny - 1 );
    dz = (zhi - zlo) / ( double ) ( nz - 1 );

    /* Set the boundary conditions. For this simple test use exact solution in bcs */

    j = 0;   // low y bc
    y = ylo;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dz;
      for ( i = 0; i < nx; i++ ) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );  // notese que cuando aca se habla de ijk, en realidad es una macro que asigna la posicion necesaria
      }
    }

    j = ny - 1;  // hi y
    y = yhi;
    
    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dz;
      for ( i = 0; i < nx; i++ ) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    i = 0;  // low x
    x = xlo;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dx;
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    i = nx - 1; // hi x
    x = xhi;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dx;    
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    k = 0; // low z
    z = zlo;

    for ( j = 0; j < ny; j++ ) {
      y = ylo + j * dy;
      for ( i = 0; i < nx; i++) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    k = nz - 1; // hi z
    z = zhi;
    
    for ( j = 0; j < ny; j++ ) {
      y = ylo + j * dy;
      for ( i = 0; i < nx; i++) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    /* Set the right hand side  */
    for ( k = 0; k < nz; k++ ){
      z = zlo + k * dz;
      for ( j = 0; j < ny; j++ ){
        y = ylo + j * dy;
        for ( i = 0; i < nx; i++ ) {
          x = xlo + i * dx;
          F(i,j,k) =  uxx_exact ( x, y, z ) + uyy_exact ( x, y, z ) + uzz_exact( x, y, z );
        }
      }
    }

    return;
}

/* ------------------------------------------------------------------------- */

void calc_error ( int nx, int ny, int nz, double u[],  double f[],
                  double xlo,  double ylo, double zlo,
                  double xhi,  double yhi, double zhi )
{
    double  error_max, error_l2;
    int     i, j, k, i_max=-1, j_max=-1, k_max = -1;
    double  u_true, u_true_norm;
    double  x, y, z, dx, dy, dz, term;

    dx = (xhi - xlo) / ( double ) ( nx - 1 );
    dy = (yhi - ylo) / ( double ) ( ny - 1 );
    dz = (yhi - ylo) / ( double ) ( nz - 1 );

    error_max   = 0.0;
    error_l2    = 0.0;

    /* print statements below are commented out but  may help in debugging */
    //printf("   i    j   k       x       y    z         uexact          ucomp            error\n");
    
    for ( k = 0; k < nz; k++ ) {
      z = zlo + k * dz;
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        for ( i = 0; i < nx; i++ ) {
            x = xlo + i * dx;
            u_true = u_exact ( x, y, z );
            term   =  U(i,j,k) - u_true;
            //printf(" %d  %d  %d %12.5e %12.5e %12.5e %12.5e  %12.5e %12.5e \n",i,j,k,x,y,z,u_true,U(i,j,k),term);
            error_l2  = error_l2 + term*term;
            if (ABS(term) > error_max){
              error_max =  ABS(term);
            }
        } /* end for i */
      } /* end for j */
    } /* end for k */

    error_l2 = sqrt(dx*dy*dz*error_l2);

    printf ( "\n  max Error in computed soln:    %12.5e    \n", error_max );
    printf ( "  l2 norm of Error on %4d by %4d by %4d grid\n  (dx %12.5e dy %12.5e dz %12.5e):   %12.5e\n",
             nx,ny,nz,dx,dy,dz,error_l2 );

    return;
}

/* ------------------------------------------------------------------------- */

double u_exact ( double x, double y, double z )
{
  double pi = 4.*atan(1.0);

  return (( 1.0 - x * x ) * ( 1.0 - y * y ) * (1.0 - z * z));  // exact solution
  //return (cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return (cos(pi*x) + cos(pi*y) + cos(pi*z));
  //return 2.;
}

/* ------------------------------------------------------------------------- */

double uxx_exact ( double x, double y, double z )
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + y ) * ( 1.0 - y ) * (1.0 - z) * (1.0 + z));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*x));
  //return 0;
}

/* ------------------------------------------------------------------------- */

double uyy_exact ( double x, double y , double z)
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + x ) * ( 1.0 - x ) * (1.0 - z) * (1.0 + z));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*y));
  //return 0;
}

/* ------------------------------------------------------------------------- */

double uzz_exact ( double x, double y , double z)
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + x ) * ( 1.0 - x ) * (1.0 - y) * (1.0 + y));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*z));
  //return 0;
}

/* ------------------------------------------------------------------------- */

void get_prob_size(int *nx, int *ny, int *nz, int argc, int* argv)
{
        /* read problem size from the arguments */
        *nx  = argv[0];
        *ny  = argv[1];
        *nz  = argv[2];
      printf("Discretizing with %d %d %d points in x y and z \n", *nx, *ny, *nz);

    return;
}

void jacobi ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval)
{

    double ax, ay, az, d;
    double dx, dy, dz;
    double update_norm, unew;
    int i, it, j, k, it_used = it_max;
    double *u_old, diff;

    /* Initialize the coefficients.  */

    dx =  (xhi - xlo) / ( double ) ( nx - 1 );
    dy =  (yhi - ylo) / ( double ) ( ny - 1 );
    dz =  (zhi - zlo) / ( double ) ( nz - 1 );

    ax =   1.0 / (dx * dx);
    ay =   1.0 / (dy * dy);
    az =   1.0 / (dz * dz);
    d  = - 2.0 / (dx * dx)  - 2.0 / (dy * dy) -2.0 / (dz * dz);

    u_old = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );

    for ( it = 1; it <= it_max; it++ ) {
        update_norm = 0.0;

        /* Copy new solution into old.  */
      for ( k = 0; k < nz; k++ ) {
        for ( j = 0; j < ny; j++ ) {
            for ( i = 0; i < nx; i++ ) {
              U_OLD(i,j,k) = U(i,j,k);
            }
        }
      }

    /* Compute stencil, and update.  bcs already in u. only update interior of domain */
      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {
            for ( i = 1; i < nx-1; i++ ) {

                unew = (F(i,j,k) -
                        ( ax * ( U_OLD(i-1,j,k) + U_OLD(i+1,j,k) ) +
                          ay * ( U_OLD(i,j-1,k) + U_OLD(i,j+1,k) ) +
                          az * ( U_OLD(i,j,k-1) + U_OLD(i,j,k+1) ) ) ) / d;

                diff = ABS(unew-U_OLD(i,j,k));
                //update_norm = update_norm + diff*diff;  /* using 2 norm */

                if (diff > update_norm){ /* using max norm */
                    update_norm = diff;
                  }

                U(i,j,k) = unew;

            } /* end for i */
        } /* end for j */
      } /* end for k */

        if (0 == it% io_interval) 
            printf ( " iteration  %5d   norm update %14.4e\n", it, update_norm );

        if ( update_norm <= tol ) {
          it_used = it;
          break;
        }

    } /* end for it iterations */


    printf ( " Final iteration  %5d   norm update %14.6e\n", it_used, update_norm );

    free ( u_old );

    return;
}

void gauss_seidel ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval, char rb [])
{

    double ax, ay, az, d;
    double dx, dy, dz, rem;
    double update_norm, unew;
    int i, it, j, k, it_used = it_max;
    double *u_old, diff;
    double omega;

    /* Initialize the coefficients.  */

    dx =  (xhi - xlo) / ( double ) ( nx - 1 );
    dy =  (yhi - ylo) / ( double ) ( ny - 1 );
    dz =  (zhi - zlo) / ( double ) ( nz - 1 );

    ax =   1.0 / (dx * dx);
    ay =   1.0 / (dy * dy);
    az =   1.0 / (dz * dz);
    d  = - 2.0 / (dx * dx)  - 2.0 / (dy * dy) -2.0 / (dz * dz);
    omega = 1;

    for ( it = 1; it <= it_max; it++ ) {
        update_norm = 0.0;

     /* Compute stencil, and update.  bcs already in u. only update interior of domain */
      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {
            for ( i = 1; i < nx-1; i++ ) {
		if('r' == RB(i,j,k)){
			rem = U(i,j,k);
                	U(i,j,k) = (F(i,j,k) -
                        	( ax * ( U(i-1,j,k) + U(i+1,j,k) ) +
                        	  ay * ( U(i,j-1,k) + U(i,j+1,k) ) +
                        	  az * ( U(i,j,k-1) + U(i,j,k+1) ) ) ) / d;

                	diff = ABS(U(i,j,k)-rem);
                
			//update_norm = update_norm + diff*diff;  /* using 2 norm */

                	if (diff > update_norm){ /* using max norm */
                	    update_norm = diff;
                	  }

               }
            } /* end for i */
        } /* end for j */
      } /* end for k */

      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {
            for ( i = 1; i < nx-1; i++ ) {
		if('b' == RB(i,j,k)){
			rem = U(i,j,k);
                	U(i,j,k) = (F(i,j,k) -
                        	( ax * ( U(i-1,j,k) + U(i+1,j,k) ) +
                        	  ay * ( U(i,j-1,k) + U(i,j+1,k) ) +
                        	  az * ( U(i,j,k-1) + U(i,j,k+1) ) ) ) / d;

                	diff = ABS(U(i,j,k)-rem);
                
			//update_norm = update_norm + diff*diff;  /* using 2 norm */

                	if (diff > update_norm){ /* using max norm */
                	    update_norm = diff;
                	  }

               }
            } /* end for i */
        } /* end for j */
      } /* end for k */

        if (0 == it% io_interval) 
            printf ( " iteration  %5d   norm update %14.4e\n", it, update_norm );

        if ( update_norm <= tol ) {
          it_used = it;
          break;
        }

    } /* end for it iterations */
    printf ( " Final iteration  %5d   norm update %14.6e\n", it_used, update_norm );

    return;
}
/* ------------------------------------------------------------------------- */

#undef U_OLD
#undef U
#undef F
#undef ABS
#undef MAX
#undef INDEX3D
