#ifndef PLOT_H
#define PLOT_H

#include <vector>
#include <cmath>
#include <string>
#include <boost/tuple/tuple.hpp>

#include "gnuplot-iostream.h"

const double PI = 3.14159265;

typedef std::pair<double, double> point;

class Plot {

    public:
        Plot();
        ~Plot();

        void clear();
        void plot(std::vector<point> & p,std::string color);
        void show();
        
        void set_x_limit(double xmin, double xmax);
        void set_y_limit(double ymin, double ymax);

        void draw_circle(double x, double y, double radius, std::string color);

        static std::string getColor(const int value);



    private:
        Gnuplot gp;
        std::vector<std::vector<point>> points;
        std::vector<std::string> colors;
        double x_min, x_max, y_min, y_max;
};

#endif