#include "../header/tools.hpp"

// If True its little endian else is big endian.
bool checkByteOrder() {
    short int word = 0x0001;
    char* b = (char *)&word;
    return b[0] == 1;
}

int reverseInt (int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}