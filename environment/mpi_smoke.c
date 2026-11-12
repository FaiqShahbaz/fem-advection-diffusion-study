/* Environment test: collective integer sum, with one or two MPI processes. */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Process r contributes r+1; every process should receive 1+...+size. */
    const int local_value = rank + 1;
    int global_sum = 0;
    MPI_Allreduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int expected = size * (size + 1) / 2;
    const int failed = global_sum != expected;

    printf("rank=%d size=%d local=%d global_sum=%d expected=%d status=%s\n",
           rank, size, local_value, global_sum, expected, failed ? "FAIL" : "PASS");
    fflush(stdout);
    MPI_Finalize();
    return failed;
}
