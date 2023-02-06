//
// Created by Yusuf Ganiyu on 2/3/23.
//
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>

#ifndef SMALLSCALE_SMALLSCALE_H
#define SMALLSCALE_SMALLSCALE_H

// A hash function used to hash a pair of any kind
struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);

        if (hash1 != hash2) {
            return hash1 ^ hash2;
        }

        // If hash1 == hash2, their XOR is zero.
        return hash1;
    }
};

#endif //SMALLSCALE_SMALLSCALE_H

