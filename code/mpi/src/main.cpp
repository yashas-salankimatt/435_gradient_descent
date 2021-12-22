#include "mpi_logistic_regression.h"
#include "helpers.h"
#define MASTER 0      /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

int main(int argc, char *argv[])
{
  std::string filepath;
  int dataset_size;
  int num_features;
  int num_labels;

  // Read in arguments.
  if (argc == 5)
  {
    filepath = argv[1];
    dataset_size = atoi(argv[2]);
    num_features = atoi(argv[3]);
    num_labels = atoi(argv[4]);
  }
  else
  {
    printf("\n incorrect amount of inputs.");
    exit(1);
  }

  int rank,                  /* process rank number */
      size,                  /* total number of processes */
      numworkers,            /* number of worker tasks */
      mtype,                 /* message type */
      rows,                  /* used to index batches */
      averow, extra, offset; /* used to determine rows sent to each worker */
  std::vector<std::vector<float>> features,
      labels; /* datastructures that represent input data */
  MPI_Status status;
  float features_flat[dataset_size * num_features]; /* Flat matrix for MPI communications */
  float labels_flat[dataset_size * num_labels];     /* Flat matrix for MPI communications */
  float master_init_time,
      master_send_time,
      worker_recv_time,
      worker_unflatten_time,
      whole_program_time,
      worker_main_algorithm_time; /* Timers */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  numworkers = size - 1;

  MPI_Barrier(MPI_COMM_WORLD);
  float whole_program_time_start = MPI_Wtime(); /*TIMER*/

  /*############ Create new communicator for worker threads. (Exclude master)############*/
  // Get the group or processes of the default communicator
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  // Select worker processes
  int ranks[numworkers];
  for (int i = 1; i <= numworkers; i++)
  {
    ranks[i - 1] = i;
  }
  MPI_Group worker_group;
  MPI_Group_incl(world_group, numworkers, ranks, &worker_group);
  // Create the new communicator from new group.
  MPI_Comm worker_communicator;
  MPI_Comm_create(MPI_COMM_WORLD, worker_group, &worker_communicator);
  /* ############ End create new communicator for worker threads. (Exclude master)############*/

  // Master process.
  if (rank == MASTER)
  {

    float master_init_time_start = MPI_Wtime(); /*TIMER*/

    // Read in input data.
    read_csv(filepath, features, labels, dataset_size);

    // Create flat arrays for mpi transfer.
    flattenVectorMatrix(features, features_flat);
    flattenVectorMatrix(labels, labels_flat);

    master_init_time = MPI_Wtime() - master_init_time_start; /*TIMER*/

    // Extra debugging information stdout.
    if (DEBUG)
    {
      std::cout << "################Feature Vector:################" << std::endl;
      printVectorMatrix(features);
      std::cout << "################End Feature Vector:################" << std::endl;
      std::cout << "################Labels Vector:################" << std::endl;
      printVectorMatrix(labels);
      std::cout << "################End Labels Vector:################" << std::endl;
      std::cout << "################Flat Feature Array:################" << std::endl;
      printFlatMatrix(features_flat, dataset_size, num_features);
      std::cout << "################End Flat Feature Array:################" << std::endl;
      std::cout << "################Flat Labels Array:################" << std::endl;
      printFlatMatrix(labels_flat, dataset_size, num_labels);
      std::cout << "################End Flat Labels Array:################" << std::endl;
    }

    float master_send_time_start = MPI_Wtime(); /*TIMER*/
    /*** Master send ***/
    averow = dataset_size / numworkers;
    extra = dataset_size % numworkers;
    offset = 0;
    mtype = FROM_MASTER;
    for (unsigned int dest = 1; dest <= numworkers; dest++)
    {
      rows = (dest <= extra) ? averow + 1 : averow;
      if (DEBUG)
        printf("Sending %d rows to task %d offset=%d\n", rows, dest, offset);
      MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&features_flat[offset], rows * num_features, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
      MPI_Send(&labels_flat[offset], rows * num_labels, MPI_FLOAT, dest, mtype, MPI_COMM_WORLD);
      offset = offset + rows;
    }
    /*** End master send ***/
    master_send_time = MPI_Wtime() - master_send_time_start; /*TIMER*/
  }

  // Worker process.
  if (rank > MASTER)
  {

    float worker_recv_time_start = MPI_Wtime(); /*TIMER*/
    /*** Worker receive ***/
    mtype = FROM_MASTER;
    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&features_flat, rows * num_features, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&labels_flat, rows * num_labels, MPI_FLOAT, MASTER, mtype, MPI_COMM_WORLD, &status);
    /*** End worker receive ***/
    worker_recv_time = MPI_Wtime() - worker_recv_time_start; /*TIMER*/

    float worker_unflatten_time_start = MPI_Wtime(); /*TIMER*/
    // Unflatten matricies to be used in computation
    std::vector<std::vector<float>> features_unflatten,
        labels_unflatten;
    for (int i = 0; i < rows; i++)
    {
      std::vector<float> labels_i = {labels_flat[i]};
      labels_unflatten.push_back(labels_i);
      std::vector<float> features_j;
      for (int j = 0; j < num_features; j++)
      {
        features_j.push_back(features_flat[num_features * i + j]);
      }
      features_unflatten.push_back(features_j);
    }
    worker_unflatten_time = MPI_Wtime() - worker_unflatten_time_start; /*TIMER*/

    float worker_main_algorithm_time_start = MPI_Wtime(); /*TIMER*/
    /***** Start main algorithm *****/
    // Initialization
    float learning_rate = 0.001;
    int iterations = 0;
    float weights_avg[num_features];
    std::vector<std::vector<float>> weights(std::vector<std::vector<float>>(num_features, std::vector<float>(1)));
    for (int i = 0; i < num_features; i++)
    { // initialize _weight vector, where -1 <= w_i <= 1
      weights[i][0] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5;
    }
    LogisticRegression *logreg = new LogisticRegression(features_unflatten, labels_unflatten, learning_rate, rank, weights);
    // Main loop. Arbitrary number of training iterations. Increase for accuracy.
    while (iterations < 30)
    {
      if (DEBUG)
        std::cout << "-----------------Training...-----------------" << std::endl;
      weights = logreg->train(weights);
      if (DEBUG)
        std::cout << "-----------------End Training-----------------" << std::endl;

      // Need all training for the current iteration to be complete.
      MPI_Barrier(worker_communicator);

      // Reduce weight values into an aggregate.
      float flat_weights[num_features];
      flattenVectorMatrix(weights, flat_weights);
      MPI_Reduce(flat_weights, weights_avg, num_features, MPI_FLOAT, MPI_SUM, 0, worker_communicator); // Note: root 0 is worker 1.

      // Distribute average of weight values back to all worker processes.
      if (rank == 1) // Worker root.
      {
        for (int i = 0; i < num_features; i++)
        {
          weights_avg[i] /= numworkers; // turn sum into avg.
        }
        MPI_Bcast(weights_avg, num_features, MPI_FLOAT, 0, worker_communicator); // Update weights_avg for all other workers
      }

      // update weights with weights_avg for next iteration.
      for (int i = 0; i < num_features; i++)
      {
        std::vector<float> weights_i = {flat_weights[i]};
        weights.at(i) = weights_i;
      }

      iterations++;
    }
    /***** End main algorithm *****/
    worker_main_algorithm_time = MPI_Wtime() - worker_main_algorithm_time_start; /*TIMER*/
  }

  MPI_Barrier(MPI_COMM_WORLD);
  whole_program_time = MPI_Wtime() - whole_program_time_start;

  /***Start Timer Collection***/
  // Buffer for info output.
  std::vector<std::string> out_buffer;
  out_buffer.push_back("MPI");
  out_buffer.push_back(",");
  out_buffer.push_back(std::to_string(size));
  out_buffer.push_back(",");
  out_buffer.push_back(std::to_string(dataset_size));
  out_buffer.push_back(",");
  out_buffer.push_back(std::to_string(num_features));
  out_buffer.push_back(",");
  out_buffer.push_back(std::to_string(num_labels));
  out_buffer.push_back(",");

  float worker_recv_time_max,
      worker_recv_time_min,
      worker_recv_time_sum,
      worker_unflatten_time_max,
      worker_unflatten_time_min,
      worker_unflatten_time_sum,
      worker_main_algorithm_time_max,
      worker_main_algorithm_time_min,
      worker_main_algorithm_time_sum;

  if (worker_communicator != MPI_COMM_NULL)
  { // Worker processes. Note: This is reducing to "worker_communicator" rank 0, which is actually rank 1 in "MPI_COMM_WORLD".
    // Worker Receive times.
    MPI_Reduce(&worker_recv_time, &worker_recv_time_max, 1, MPI_FLOAT, MPI_MAX, 0, worker_communicator);
    MPI_Reduce(&worker_recv_time, &worker_recv_time_min, 1, MPI_FLOAT, MPI_MIN, 0, worker_communicator);
    MPI_Reduce(&worker_recv_time, &worker_recv_time_sum, 1, MPI_FLOAT, MPI_SUM, 0, worker_communicator);
    // Worker Unflatten times.
    MPI_Reduce(&worker_unflatten_time, &worker_unflatten_time_max, 1, MPI_FLOAT, MPI_MAX, 0, worker_communicator);
    MPI_Reduce(&worker_unflatten_time, &worker_unflatten_time_min, 1, MPI_FLOAT, MPI_MIN, 0, worker_communicator);
    MPI_Reduce(&worker_unflatten_time, &worker_unflatten_time_sum, 1, MPI_FLOAT, MPI_SUM, 0, worker_communicator);
    // Worker Algorithm times.
    MPI_Reduce(&worker_main_algorithm_time, &worker_main_algorithm_time_max, 1, MPI_FLOAT, MPI_MAX, 0, worker_communicator);
    MPI_Reduce(&worker_main_algorithm_time, &worker_main_algorithm_time_min, 1, MPI_FLOAT, MPI_MIN, 0, worker_communicator);
    MPI_Reduce(&worker_main_algorithm_time, &worker_main_algorithm_time_sum, 1, MPI_FLOAT, MPI_SUM, 0, worker_communicator);
  }

  if (rank == 0)
  { // Master times.
    out_buffer.push_back(std::to_string(whole_program_time));
    out_buffer.push_back(",");
    out_buffer.push_back(std::to_string(master_init_time));
    out_buffer.push_back(",");
    out_buffer.push_back(std::to_string(master_send_time));
    out_buffer.push_back(",");
    for (unsigned int i = 0; i < out_buffer.size(); i++)
    {
      std::cout << out_buffer.at(i);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 1)
  { // Output worker results only once.
    // Buffer for worker info output.
    std::vector<std::string> worker_out_buffer;
    worker_out_buffer.push_back(std::to_string(worker_recv_time_max));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_recv_time_min));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_recv_time_sum / (float)numworkers));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_unflatten_time_max));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_unflatten_time_min));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_unflatten_time_sum / (float)numworkers));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_main_algorithm_time_max));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_main_algorithm_time_min));
    worker_out_buffer.push_back(",");
    worker_out_buffer.push_back(std::to_string(worker_main_algorithm_time_sum / (float)numworkers));
    for (unsigned int i = 0; i < worker_out_buffer.size(); i++)
    {
      std::cout << worker_out_buffer.at(i);
    }
  }
  /***End Timer Collection***/

  MPI_Finalize();
  return 0;
}