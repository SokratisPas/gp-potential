#pragma once

#include <vector>
#include <array>
#include <filesystem>

#include "mpi.h"

#include "individual.h"

// --------------------------------------------------------
class GlobalHOF
{
public:
    // struct for serialized individuals to send with MPI
    struct SerializedEntry
    {
        double fitness = std::numeric_limits<double>::infinity();
        std::array<int, 4> sizes{};
        std::array<int, 4> depths{};
        std::array<std::string, 4> trees{};
    };

    std::vector<SerializedEntry> gatheredHof;  // gathered local HOFs in global hof

    static std::string SerializeEntry(const SerializedEntry& entry);
    static std::vector<char> SerializeLocalEntries(const std::vector<SerializedEntry>& localEntries);
    static std::vector<SerializedEntry> DeserializeLocalEntries(const std::vector<char>& buffer,
                                                              size_t expectedCount);

    void GatherLocalHOF(int rank,
                       int worldSize,
                       int NlocalInds,
                       const std::vector<Individual>& localHof);

    void writeGlobalHOF(int rank,
        const std::filesystem::path& outputDir = "output");  // root writes the gathered HOF
};


// ================================
std::string GlobalHOF::SerializeEntry(const SerializedEntry& entry)
{
    std::string buffer;
    buffer.reserve(sizeof(double) + 4 * sizeof(int) + 4 * sizeof(int) + 4 * 4 * sizeof(uint32_t));

    auto appendValue = [&buffer](const auto& value)
    {
        const char* data = reinterpret_cast<const char*>(&value);
        buffer.append(data, sizeof(value));
    };

    appendValue(entry.fitness);

    for (const auto& value : entry.sizes)
        appendValue(value);

    for (const auto& value : entry.depths)
        appendValue(value);

    for (const auto& tree : entry.trees)
    {
        const uint32_t len = static_cast<uint32_t>(tree.size());
        appendValue(len);
        buffer.append(tree);
    }

    return buffer;
}

// ================================
std::vector<char> GlobalHOF::SerializeLocalEntries(const std::vector<SerializedEntry>& localEntries)
{
    std::vector<char> buffer;
    size_t totalSize = 0;

    for (const auto& entry : localEntries)
    {
        const std::string serialized = SerializeEntry(entry);
        totalSize += serialized.size();
    }

    buffer.reserve(totalSize);

    for (const auto& entry : localEntries)
    {
        const std::string serialized = SerializeEntry(entry);
        buffer.insert(buffer.end(), serialized.begin(), serialized.end());
    }

    return buffer;
}

// ================================
std::vector<GlobalHOF::SerializedEntry> GlobalHOF::DeserializeLocalEntries(const std::vector<char>& buffer,
                                                                        size_t expectedCount)
{
    std::vector<SerializedEntry> entries;
    entries.reserve(expectedCount);

    size_t pos = 0;
    for (size_t i = 0; i < expectedCount; ++i)
    {
        SerializedEntry entry{};

        auto readValue = [&buffer, &pos](auto& value)
        {
            if (pos + sizeof(value) > buffer.size())
                throw std::runtime_error("Serialized HOF entry is truncated");

            std::memcpy(&value, buffer.data() + pos, sizeof(value));
            pos += sizeof(value);
        };

        readValue(entry.fitness);

        for (auto& value : entry.sizes)
            readValue(value);

        for (auto& value : entry.depths)
            readValue(value);

        for (auto& tree : entry.trees)
        {
            uint32_t len = 0;
            readValue(len);

            if (pos + len > buffer.size())
                throw std::runtime_error("Serialized HOF tree string is truncated");

            tree.assign(buffer.data() + pos, buffer.data() + pos + len);
            pos += len;
        }

        entries.push_back(entry);
    }

    return entries;
}


// ================================
void GlobalHOF::GatherLocalHOF(
    int rank,
    int worldSize,
    int NlocalInds,
    const std::vector<Individual>& localHof)
{
    // ------------------------------------------
    // build the serialized local entries
    std::vector<SerializedEntry> localEntries(static_cast<size_t>(NlocalInds));

    for (int i = 0; i < NlocalInds; ++i)
    {
        if (i < static_cast<int>(localHof.size()))
        {
            const auto& ind = localHof[static_cast<size_t>(i)];
            localEntries[static_cast<size_t>(i)].fitness = ind.fitness; // save fitness

            // save trees, sizes, depths
            for (int j = 0; j < 4; ++j)
            {
                localEntries[static_cast<size_t>(i)].trees[j] = ind.trees[j].convertToStr();
                localEntries[static_cast<size_t>(i)].sizes[j] = ind.trees[j].Size();
                localEntries[static_cast<size_t>(i)].depths[j] = ind.trees[j].Depth();
            }
        }
        else 
        {
            throw std::runtime_error("Local HOF size is less than NlocalInds");
        }
    }

    // ------------------------------------------
    // gather the byte counts of the sendBuffer from all ranks to rank 0
    const std::vector<char> sendBuffer = SerializeLocalEntries(localEntries);   // serialized individuals to send
    const int sendCount = static_cast<int>(sendBuffer.size());

    std::vector<int> recvCounts(static_cast<size_t>(worldSize), 0);
    std::vector<int> displacements(static_cast<size_t>(worldSize), 0);

    MPI_Gather(
        &sendCount,         // const void *sendbuf
        1,                  // int sendcount
        MPI_INT,            // MPI_Datatype sendtype
        recvCounts.data(),  // void *recvbuf
        1,                  // int recvcount
        MPI_INT,            // MPI_Datatype recvtype
        0,                  // int root
        MPI_COMM_WORLD      // MPI_Comm comm
    );

    // ------------------------------------------
    // gather the data (now we know the size of each sendBuffer)
    if (rank == 0)
    {
        int totalRecv = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            displacements[i] = totalRecv;
            totalRecv += recvCounts[i];
        }

        std::vector<char> recvBuffer(static_cast<size_t>(totalRecv));
        const char* sendPtr = sendCount > 0 ? sendBuffer.data() : nullptr;

        // MPI_Gatherv to send different sized buffers from each rank to rank 0
        MPI_Gatherv(
            const_cast<char*>(sendPtr),     // const void *sendbuf
            sendCount,                      // int sendcount
            MPI_CHAR,                       // MPI_Datatype sendtype
            recvBuffer.data(),              // void *recvbuf
            recvCounts.data(),              // const int recvcounts[]
            displacements.data(),           // const int displs[]
            MPI_CHAR,                       // MPI_Datatype recvtype
            0,                              // int root
            MPI_COMM_WORLD                  // MPI_Comm comm
        );

        gatheredHof = DeserializeLocalEntries(recvBuffer, 
            static_cast<size_t>(worldSize) * static_cast<size_t>(NlocalInds));
    }
    else
    {
        const char* sendPtr = sendCount > 0 ? sendBuffer.data() : nullptr;
        MPI_Gatherv(
            const_cast<char*>(sendPtr), // const void *sendbuf
            sendCount,                  // int sendcount
            MPI_CHAR,                   // MPI_Datatype sendtype
            nullptr,                    // void *recvbuf
            nullptr,                    // const int recvcounts[]
            nullptr,                    // const int displs[]
            MPI_CHAR,                   // MPI_Datatype recvtype
            0,                          // int root
            MPI_COMM_WORLD              // MPI_Comm comm
        );
    }
}

// ================================
void GlobalHOF::writeGlobalHOF(int rank, const std::filesystem::path& outputDir)
{
    // only rank 0 writes the globalHof file
    if (rank != 0)
        return;

    const std::filesystem::path outputDirMaster = std::filesystem::absolute(outputDir);
    std::filesystem::create_directories(outputDirMaster);

    const std::filesystem::path hofFilePathMaster = outputDirMaster / "global_hof.txt";
    std::ofstream hofFileGlobal(hofFilePathMaster, std::ios::trunc);
    if (!hofFileGlobal)
    {
        throw std::runtime_error("Failed to open output file: " + hofFilePathMaster.string());
    }

    for (const auto& entry : gatheredHof)
    {
        hofFileGlobal << "=====================\n"
            << "Fitness : " << entry.fitness << "\n"
            << "Size : " << entry.sizes[0] << ", " << entry.sizes[1] << ", "
            << entry.sizes[2] << ", " << entry.sizes[3] << "\n"
            << "Depth : " << entry.depths[0] << ", " << entry.depths[1] << ", "
            << entry.depths[2] << ", " << entry.depths[3] << "\n"
            << "Potentials : \n";

        for (int i = 0; i < 4; ++i)
        {
            hofFileGlobal << i + 1 << ") " << entry.trees[i].data() << "\n";
        }
    }

    hofFileGlobal.close();
}