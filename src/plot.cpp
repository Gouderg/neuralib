#include "../header/plot.hpp"

// Constructor.
Plot::Plot() {
    this->points.reserve(1);
    this->colors.reserve(1);
}

// Desctructor.
Plot::~Plot() {
    this->points.resize(0);
    this->colors.resize(0);
    gp.clearTmpfiles();
}

void Plot::clear() {
    this->points.resize(0);
    this->colors.resize(0);
    gp.clearTmpfiles();    
}

void Plot::set_x_limit(double xmin, double xmax) {
    this->x_min = xmin;
    this->x_max = xmax;
}

void Plot::set_y_limit(double ymin, double ymax) {
    this->y_min = ymin;
    this->y_max = ymax;
}

void Plot::set_legend(const std::string xlabel, const std::string ylabel, const std::string title) {
    this->xlabel = xlabel;
    this->ylabel = ylabel;
    this->title = title;
}


void Plot::draw_circle(double x, double y, double radius, std::string color) {
    std::vector<point> p;
    p.push_back(point(x, y));
    this->points.push_back(p);
    this->colors.push_back(color);
}

void Plot::draw_line(std::vector<double> y, std::string color) {
    gp << "set yrange [" << this->y_min << ":" << this->y_max << "]\n";
    gp << "set xrange [" << this->x_min << ":" << this->x_max << "]\n";
    gp << "set xlabel '" << this->xlabel << "'\n";
    gp << "set ylabel '" << this->ylabel << "'\n";
    gp << "set title '" << this->title << "'\n";


    gp << "set linetype 1 lc rgb '" << color << "' lw 2 pt 1\n";

    for (int i = 0; i < y.size(); i++) {
        if (i != y.size()-1) {
            gp << " '-' with linespoints 1, ";    
        } else {
            gp << " '-' with linespoints 1\n";    
        }
    }

    gp.send1d(y);

    gp << "unset xlabel\n";
    gp << "unset ylabel\n";

}

void Plot::setMultiplot(const int row, const int column) {
    gp.clearTmpfiles();
    gp << "set multiplot layout "<< row << "," << column << " columnsfirst \n";
}

void Plot::unsetMultiplot() {
    gp << "unset multiplot\n";
}


std::string Plot::getColor(const int value) {
    switch (value) {
        default: case 0:
            return "'blue'";

        case 1:
            return "'red'";
        
        case 2:
            return "'green'";
        
        case 3:
            return "'yellow'";
    }
}

void Plot::show() {
    gp.clearTmpfiles();
    gp << "set yrange [" << this->y_min << ":" << this->y_max << "]\n";
    gp << "set xrange [" << this->x_min << ":" << this->x_max << "]\n";
        
    gp<<"plot";

    for (int i = 0; i < this->colors.size(); i++) {
        if (i != this->colors.size() - 1) {
            gp << " '-' with points pt 7 ps 1 lc " << this->colors[i] << " notitle,";
        } else {
            gp << " '-' with points pt 7 ps 1 lc " << this->colors[i] << " notitle\n";
        }
    }

    for (int i = 0; i < this->colors.size(); i++) {
        gp.send1d(this->points[i]);
    }

}