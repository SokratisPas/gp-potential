#pragma once

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

struct GPConfig {
    int popSize = 100;
    int gens = 50;
    int initialIndMaxDepth = 5;
    int tournamentSize = 3;
    double mutationProb = 0.2;
    double constMin = -10.0;
    double constMax = 10.0;
    int NdataSample = 50;
    int lastNdata = 100;
    double cutoff = 5.0;
    int NlocalInds = 2;
    int NgensToSendInds = 10;
    int NgensToMigration = 20;
    std::string dataPath = "src/data/W-Mo/trainset.xyz";
    std::filesystem::path outputDir = "run/output";
};

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";

    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::map<std::string, std::string> parseConfigFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Failed to open config file: " + path.string());

    std::map<std::string, std::string> values;
    std::string line;
    int lineNumber = 0;

    while (std::getline(file, line))
    {
        ++lineNumber;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        const auto pos = trimmed.find('=');
        if (pos == std::string::npos)
            throw std::runtime_error("Invalid config line " + std::to_string(lineNumber) + ": " + line);

        const std::string key = trim(trimmed.substr(0, pos));
        const std::string value = trim(trimmed.substr(pos + 1));
        if (key.empty())
            throw std::runtime_error("Empty config key on line " + std::to_string(lineNumber));

        values[key] = value;
    }

    return values;
}

std::string requireValue(const std::map<std::string, std::string>& values, const std::string& key)
{
    const auto it = values.find(key);
    if (it == values.end())
        throw std::runtime_error("Missing config key: " + key);
    return it->second;
}

int parseInt(const std::map<std::string, std::string>& values, const std::string& key)
{
    std::stringstream ss(requireValue(values, key));
    int value = 0;
    ss >> value;
    if (!ss || !ss.eof())
        throw std::runtime_error("Invalid integer value for " + key);
    return value;
}

double parseDouble(const std::map<std::string, std::string>& values, const std::string& key)
{
    std::stringstream ss(requireValue(values, key));
    double value = 0.0;
    ss >> value;
    if (!ss || !ss.eof())
        throw std::runtime_error("Invalid number value for " + key);
    return value;
}

GPConfig loadConfig(const std::filesystem::path& configPath)
{
    const auto values = parseConfigFile(configPath);
    GPConfig config;

    config.popSize = parseInt(values, "pop_size");
    config.gens = parseInt(values, "gens");
    config.initialIndMaxDepth = parseInt(values, "initial_ind_max_depth");
    config.tournamentSize = parseInt(values, "tournament_size");
    config.mutationProb = parseDouble(values, "mutation_prob");
    config.constMin = parseDouble(values, "const_min");
    config.constMax = parseDouble(values, "const_max");
    config.NdataSample = parseInt(values, "ndata_sample");
    config.lastNdata = parseInt(values, "last_n_data");
    config.cutoff = parseDouble(values, "cutoff");
    config.NlocalInds = parseInt(values, "n_local_inds");
    config.NgensToSendInds = parseInt(values, "ngens_to_send_inds");
    config.NgensToMigration = parseInt(values, "ngens_to_migration");
    config.dataPath = requireValue(values, "data_path");

    if (values.count("output_dir") != 0)
        config.outputDir = values.at("output_dir");

    return config;
}

void printUsage(const char* programName)
{
    std::cout << "Usage: " << programName << " [--config path] [--output-dir path]\n";
    std::cout << "Defaults:\n";
    std::cout << "  --config /input.txt\n";
    std::cout << "  --output-dir /output\n";
}