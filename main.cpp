#include <string>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <set>

using namespace std;
int main(int argc, char* argv[]){
    int rank; // rank of the current processor
    int size; // total number of processors

    int P=0;  // total number of processors
    int N=0;  // number of instances
    int A=0;  // number of features
    int M=0;  // iteration count
    int T=0;  // resulting number of features

    string input_arg= argv[1];
    ifstream input_file;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // gets the rank of the current processor
    MPI_Comm_size(MPI_COMM_WORLD, &size); // gets the total number of processors

    int masterSignal=0;

    if(rank == 0){
        input_file.open(input_arg);
        input_file>>P>>N>>A>>M>>T;
    }
    // Master sends M and T to the slaves
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int input_size = N*(A+1);
    float input[input_size];

    int slave_input_size= input_size / (P-1);
    float slave_input[slave_input_size];

    float scatter[slave_input_size*P];

    float weights[A];
    for(int i=0; i<A; i++)
        weights[i]=0;

    int arr[P*T];   // chosen features
    for(int i = 0; i<P*T; i++)
        arr[i]= -1;

    int res[T];    // chosen features by each slave
    for(int i = 0; i<T; i++)
        res[i]= -1;

    if(rank==0){
        float num;
        for(int i=0;i<input_size;i++){
            input_file>>num;
            input[i]= num;
        }
        input_file.close();

        // MPI_Scatter also sends to root itself, so first partition is filled with dummy data, rest of input is copied
        for(int i=0; i<slave_input_size; i++)
            scatter[i]=0;
        for(int i=0; i<input_size; i++){
            scatter[slave_input_size+i]= input[i];
        }
    }

    // Master partitions the input between different processors
    MPI_Scatter(scatter,slave_input_size,MPI_FLOAT,
                slave_input,slave_input_size,MPI_FLOAT,
                0,MPI_COMM_WORLD);

    masterSignal = 1;
    while(masterSignal){
        if(rank!= 0){
            for(int i=0; i<M; i++){

                int target = i*(A+1);
                bool target_flag = slave_input[target+A];   // class variable of target instance

                float manhattan[N/(P-1)];   // manhattan array holds the distance between target and each instance
                float max[A];   // maximum value of each feature
                float min[A];   // minimum value of each feature
                for(int j=0; j<A; j++){
                    max[j]=slave_input[j];
                    min[j]=slave_input[j];
                }

                float nearest_hit_distance = INT_MAX;
                int nearest_hit = i;
                float nearest_miss_distance = INT_MAX;
                int nearest_miss = i;

                for(int j=0; j<N/(P-1); j++){
                    float distance = 0;
                    for(int k=0; k<A; k++){
                        float kth_feature = slave_input[j*(A+1)+k];
                        if(kth_feature<min[k])
                            min[k] = kth_feature;
                        if(kth_feature>max[k])
                            max[k] = kth_feature;
                        distance+=abs(slave_input[target+k] - slave_input[j*(A+1)+k]);
                    }

                    bool flag = slave_input[j*(A+1)+A]; // class variable of current instance
                    manhattan[j]=distance; // sum of absolute differences of corresponding features between target and current instance

                    // Checks if this instance can be the nearest hit or nearest miss
                    if(j!=i){
                        if(target_flag==flag && distance<nearest_hit_distance){
                            nearest_hit_distance = distance;
                            nearest_hit = j;
                        }
                        else if(target_flag!=flag && distance<nearest_miss_distance) {
                            nearest_miss_distance = distance;
                            nearest_miss = j;
                        }
                    }
                }

                // Update weights according to Relief algorithm
                for(int a=0; a<A; a++){
                    weights[a] = weights[a]
                                 - (abs(slave_input[target+a]-slave_input[nearest_hit*(A+1)+a]) / (max[a]-min[a])*1.0)/M*1.0
                                 + (abs(slave_input[target+a]-slave_input[nearest_miss*(A+1)+a]) / (max[a]-min[a])*1.0)/M*1.0;
                }
            }

            // Select top T features
            set<int, greater<int> > mySet;
            for(int j=0; j<T; j++){
                float max = INT_MIN;
                int max_index = -1;
                for(int i=0; i<A; i++){
                    if(weights[i]>=max && mySet.find(i)==mySet.end()){
                        max = weights[i];
                        max_index = i;
                    }
                }
                mySet.insert(max_index);
                res[j]= max_index;
            }

            // Each slave prints output in ascending order
            sort(res, res+T);
            string output = "Slave P" + to_string(rank) + " : ";
            for(int i=0; i<T; i++){
                output.append(to_string(res[i]) + " ");
            }
            output.pop_back();
            cout<<output<<endl;
        }

        // The root gathers the top selections of all slaves into an array
        MPI_Gather(&res, T, MPI_INT,
                   arr, T, MPI_INT,
                   0, MPI_COMM_WORLD);

        // The master unites the selected features in a set and prints in  ascending order
        if(rank == 0){
            masterSignal=0;
            set<int, greater<int> > result_set;
            set<int>::reverse_iterator itr;

            for(int i=T; i<P*T ; i++){
                result_set.insert(arr[i]);
            }
            string master_output = "Master P0 : ";
            for(itr = result_set.rbegin(); itr!=result_set.rend(); itr++)
                master_output.append(to_string(*itr)+ " ");
            master_output.pop_back();
            cout<<master_output<<endl;
        }

        MPI_Bcast(&masterSignal, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
