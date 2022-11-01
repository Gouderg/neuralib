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

void Plot::draw_circle(double x, double y, double radius, std::string color) {
    draw_ellipse(x,y,radius,radius,color);
}

void Plot::draw_ellipse(double x, double y, double a, double b, std::string color) {
    double x_, y_;
    std::vector<point> p;

    for (int i = 0; i < 360; i++){
        x_ = a*cos(i * PI / 180) + x;
        y_ = b*sin(i * PI / 180) + y;
        p.push_back(point(x_, y_));
    }
    this->points.push_back(p);
    this->colors.push_back(color);
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


void Plot::plot(std::vector<point> & p,std::string color) {
    this->points.push_back(p);
    this->colors.push_back(color);
}

void Plot::show() {
    gp.clearTmpfiles();
    gp << "set yrange [" << this->y_min << ":" << this->y_max << "]\n";
    gp << "set xrange [" << this->x_min << ":" << this->x_max << "]\n";
    
    for (int i = 0; i < this->colors.size(); i++) {
        gp << "set linetype " << i + 1 <<" linecolor rgb "<< this->colors[i]<<std::endl;
    }

    
    gp<<"plot";

    for (int i = 0; i < this->colors.size(); i++) {
        if (i != this->colors.size() - 1) {
            gp << " '-' with linespoint ls " << i + 1 << " points 0 notitle,";
        } else {
            gp << " '-' notitle with linespoint ls " << i + 1 << " points 0\n";
        }
    }

    for (int i = 0;i < this->colors.size(); i++) {
        gp.send1d(this->points[i]);
    }

}