#ifndef PLOT_H
#define PLOT_H

#include <vector>
#include <cmath>
#include <string>
#include <boost/tuple/tuple.hpp>

#include "../libs/gnuplot-iostream.h"

const double PI = 3.14159265;

typedef std::pair<double, double> point;

class Plot {

    public:
        Plot();
        ~Plot();

        void clear();
        void show();
        
        void set_x_limit(double xmin, double xmax);
        void set_y_limit(double ymin, double ymax);
        void set_legend(const std::string xlabel, const std::string ylabel, const std::string title);

        void draw_circle(double x, double y, std::string color);
        void draw_line(std::vector<double> y, std::string color);

        static std::string getColor(const int value);

        void setMultiplot(const int row, const int column);
        void unsetMultiplot();

    private:
        Gnuplot gp;
        std::vector<std::vector<point>> points;
        std::vector<std::string> colors;
        double x_min, x_max, y_min, y_max;
        std::string xlabel, ylabel, title;
};

#endif