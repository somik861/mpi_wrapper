#include <cassert>
#include <compare>
#include <fmt/core.h>
#include <iostream>
#include <mpi.h>
#include <random>
#include <set>
#include <sstream>
#include <vector>

#define print(...) std::cout << fmt::format(__VA_ARGS__) << std::endl;

int to_int(std::string_view str) {
    std::stringstream ss;
    ss.str(std::string(str));
    int num;
    ss >> num;
    return num;
}

class Walker {
  public:
    int start_pos;
    int pos;
    int remaining_steps;

    friend auto operator<=>(const Walker&, const Walker&) = default;
};
MPI_Datatype MPI_Walker;

enum class tag { finish, continue_, print };

void print_walker(const Walker& w, int min, int max) {
    std::string to_print = "[";

    for (int i = min; i <= max; ++i) {
        if (i == w.start_pos)
            to_print += "X";
        else if (i == w.pos)
            to_print += "O";
        else if (w.start_pos < w.pos && w.start_pos < i && i < w.pos)
            to_print += "#";
        else if (w.pos < w.start_pos && (i < w.pos || w.start_pos < i))
            to_print += "#";
        else
            to_print += "_";
    }

    to_print += "]";
    std::cout << to_print << std::endl;
}

std::pair<std::vector<Walker>, tag> recieve_walkers() {
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
    int count;
    MPI_Get_count(&stat, MPI_Walker, &count);

    std::vector<Walker> data(count);
    MPI_Recv(data.data(), count, MPI_Walker, stat.MPI_SOURCE, stat.MPI_TAG,
             MPI_COMM_WORLD, &stat);

    MPI_Get_count(&stat, MPI_Walker, &count);
    assert(data.size() == count);
    return {data, static_cast<tag>(stat.MPI_TAG)};
}

void send_walkers(const std::vector<Walker>& walkers, int recv, tag tag_) {
    MPI_Send(walkers.data(), walkers.size(), MPI_Walker, recv,
             static_cast<int>(tag_), MPI_COMM_WORLD);
}

void walk(int min, int max) {
    MPI_Init(nullptr, nullptr);
    int world_rank, world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int next_rank = (world_rank + 1) % world_size;

    MPI_Type_contiguous(3, MPI_INT, &MPI_Walker);
    MPI_Type_commit(&MPI_Walker);

    int range = max - min + 1;
    assert(range >= world_size);
    int part_size = range / world_size;

    int my_start = world_rank * part_size + min;
    int my_end = (world_rank + 1) * part_size - 1 + min;
    // Im the last one
    if (world_rank == world_size - 1)
        my_end = max;

    print("RANK: {}, domain: {} - {}", world_rank, my_start, my_end);

    std::vector<Walker> my_walkers;
    for (int n = 0; n < world_size; ++n)
        my_walkers.emplace_back(my_start, my_start,
                                std::random_device{}() % 100);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
        print("Starting simulation");

    std::size_t iteration = 1;
    while (true) {
        if (world_rank == 0)
            print("Iteration: {}", iteration);
        std::vector<Walker> to_send;
        bool modified = false;
        for (auto& walker : my_walkers) {
            if (walker.remaining_steps == 0)
                continue;
            modified = true;
            (++walker.pos) %= max + 1;
            --walker.remaining_steps;
            if (!(my_start <= walker.pos && walker.pos <= my_end))
                to_send.push_back(walker);
        }

        std::erase_if(my_walkers, [&](const auto& w) {
            return std::ranges::count(to_send, w);
        });

        tag send_tag = modified ? tag::continue_ : tag::finish;
        if (world_rank == 0) {
            send_walkers(to_send, next_rank, send_tag);
        }

        auto [walkers, tag_] = recieve_walkers();
        my_walkers.insert(my_walkers.end(), walkers.begin(), walkers.end());
        if (tag_ == tag::continue_ || tag_ == tag::print)
            send_tag = tag_;

        if (world_rank != 0)
            send_walkers(to_send, next_rank, send_tag);

        if (world_rank == 0 && tag_ == tag::finish) {
            send_walkers({}, next_rank, tag::print);
            recieve_walkers();
            tag_ = tag::print;
        }

        if (tag_ == tag::print) {
            for (auto& walker : my_walkers)
                print_walker(walker, min, max);
            break;
        }

        ++iteration;
    };
    MPI_Finalize();
}

int main(int argc, char** argv) {
    if (argc != 3) {
        print("./PROGRAM MIN MAX");
        return 1;
    }
    walk(to_int(argv[1]), to_int(argv[2]));
}