#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

struct Point {
    int x, y;

    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

class SparseMatrix {
public:
    SparseMatrix(int n) {
        size = n;
        row_ptrs.resize(n + 1);
        row_ptrs[0] = 0;
    }

    void insert(int i, int j, float value) {
        data.push_back(value);
        indices.push_back(Point(i, j));
        row_ptrs[indices.back().x + 1]++;
    }

    void finalize() {
        for (int i = 0; i < size; i++) {
            row_ptrs[i + 1] += row_ptrs[i];
        }
    }

    void printToPPM(const std::string& filename) {
        std::vector<unsigned char> image(size * size * 3, 255);
        bool isBlack = true;  // flag to alternate between black and white pixels
        for (int i = 0; i < size; i++) {
            for (int j = row_ptrs[i]; j < row_ptrs[i + 1]; j++) {
                int x = indices[j].y;
                int y = indices[j].x;
                float value = data[j];
                // print alternating black and white pixels
                int pixel_offset = (y * size + x) * 3;
                if (isBlack) {
                    image[pixel_offset] = 0;
                    image[pixel_offset + 1] = 0;
                    image[pixel_offset + 2] = 0;
                } else {
                    image[pixel_offset] = 255;
                    image[pixel_offset + 1] = 255;
                    image[pixel_offset + 2] = 255;
                }
                isBlack = !isBlack;
            }
            if (size % 2 == 0) isBlack = !isBlack;  // if there is an even number of columns, switch the color for the next row
        }
        std::ofstream out(filename, std::ios::binary);
        out << "P6\n" << size << " " << size << "\n255\n";
        out.write((const char*)image.data(), image.size());
        out.close();
    }


private:
    int size;
    std::vector<float> data;
    std::vector<Point> indices;
    std::vector<int> row_ptrs;
};