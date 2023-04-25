#include "../header/dataset.hpp"

Data Dataset::spiral_data(const int samples, const int classes) {
    
    // Normal distribution. 
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{MEAN, STD_DEVIATION};


    TensorInline X({samples * classes, NB_INPUTS});
    TensorInline y({1, samples * classes});

    double r = 0.0; // Radius for the angle.
    double t = 0.0, t_random = 0.0; // theta
    int color = -1;
    int cpt = 0;
    double step_T = 4.0 / static_cast<double>(samples-1);
    
    for (int i = 0; i < classes * samples * 2; i += 2) {

        if (i % (samples * 2) == 0) {
            color += 1;
            cpt = 0;
            t = color * 4.0; 
        }

        t_random = t + (d(gen)) * 0.2;
        r = cpt / static_cast<double>(samples-1);
        X.tensor[i] = r * sin(t_random*2.5);
        X.tensor[i + 1] = r * cos(t_random*2.5);
        y.tensor[static_cast<int>(i/2)] = color;
        t += step_T;
        cpt += 1;
    }
    return {.X=X, .y=y};
}

Data Dataset::sine_data(const int samples) {
    TensorInline X({samples, 1});
    TensorInline y({samples, 1});

    for (int i = 0; i < samples; i++) {
        X.tensor[i] = i / static_cast<double>(samples);
        y.tensor[i] = sin(2.0 * M_PI * X.tensor[i]);
    }

    return {.X=X, .y=y};
}


TensorInline Dataset::read_idx_file(const std::string path, const FileType fileType) {

    // Tcheck if the processor is big endian or little endian cause the file was encode in big endian.
    bool isLittleEndian = tcheckByteOrder();
    
    std::ifstream file(path, std::ios::binary | std::ios::in);

    // If error in opening;
    if (!file) { 
        std::cerr << "Error: can't open the file." << std::endl; 
        exit(0);
    }

    // First we read the magic number and tcheck if its valid.
    int32_t magic;
    file.read((char*) &magic, sizeof(magic));
    if (isLittleEndian) {
        magic = reverseInt(magic);
    }
    
    if (magic != fileType) {
        std::cerr << "Error: file is corrupted, magic number doesn't match." << std::endl; 
        exit(0);
    }
    
    // After, we read the number of images or labels 
    int32_t numberElement;
    file.read((char*) &numberElement, sizeof(numberElement));
    if (isLittleEndian) {
        numberElement = reverseInt(numberElement);
    }

    
    // If the file is for image we need to have the width and the height
    int32_t height = 1.0, width = 1.0;

    if (fileType == FileType::images) {
        file.read((char*) &height, sizeof(height));
        if (isLittleEndian) {
            height = reverseInt(height);
        }

        file.read((char*) &width, sizeof(width));
        if (isLittleEndian) {
            width = reverseInt(width);
        }
    }


    TensorInline labelsOrImage({numberElement, width * height, false, 0.0});

    unsigned char temp = 0;
    for (int i = 0; i < numberElement * width * height; i++) {
        file.read((char*)&temp,sizeof(temp));
        labelsOrImage.tensor[i] = temp;
    }

    return labelsOrImage;    
}

void Dataset::scale_pixels_values(TensorInline& X, ScaleFormat scale) {
    if (scale == ScaleFormat::between0And1) {
        X /= 255.0;
    } else if (scale == ScaleFormat::betweenMinus1And1) {
        X -= 127.5;
        X /= 127.5;
    }
}